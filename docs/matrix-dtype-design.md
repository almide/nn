# Matrix[T] と numeric dtype: 設計メモ

> 2026-04-14 作成。Almide の `Matrix` を dtype パラメトリック化する方針を、
> 汎用言語としての整合性を壊さずに決めるための議論メモ。

## 1. 動機

Whisper tiny の 1024×1024 matmul で **Almide = 236 GFLOPS** (iter-dependent bench で CSE hoist 除外後)、
NumPy (Apple Accelerate SGEMM, f32) = **1362 GFLOPS** — 約 **5.8 倍差**。

### 現状実装 (重要: 初期調査の誤認訂正)

当初「ピュア Rust 単スレッド naive tile」と記述していたが、実際は:
- `runtime/rs/burn/matrix_burn.rs` が `src/cli/mod.rs` の `replace_matrix_runtime()` で swap される
- バックエンドは **burn 0.16 + NdArray** + `blas-src` features=["accelerate"]
- つまり **既に Apple Accelerate の `cblas_dgemm` (f64) を呼んでいる**
- 236 GFLOPS はこの構成での値

### C raw ベンチで判明した天井 (2026-04-14 実測)

| 対象 | 1024² GFLOPS | 備考 |
|---|---:|---|
| C raw Accelerate `cblas_sgemm` (f32) | **~1500** | f32 理論天井 |
| NumPy 2.2 (f32) | 1362 | 天井の 91% |
| C raw Accelerate `cblas_dgemm` (f64) | **~350** | **f64 理論天井** |
| 現状 Almide (burn + Accelerate f64) | 236 | f64 天井の 68% (burn overhead 30%) |

### 真の課題

**f64 では NumPy に原理的に勝てない**。f64 dgemm 天井 350 GFLOPS は NumPy f32 (1362) の **1/4**。
- SIMD レーン: f32 は f64 の 2 倍
- Apple AMX: FP32 経路が最速、FP64 は約半分
- f64 BLAS を最適化しても NumPy f32 の 1/4 止まり

**NumPy タイに至る唯一の道は f32 対応**。そのためには言語レベルで dtype を区別する機構 (Matrix[T] + Numeric primitive types) が必須。
「場当たり的に `matrix_f32.mul` / `matrix_f64.mul` を並置するのではなく、
Almide の型システム全体と整合した形」で設計する。

## 2. 目標

1. **Almide 全体感を壊さない** — `List[T]` / `Option[T]` と同じ哲学で `Matrix[T]` を扱いたい
2. **MSR (LLM 書きやすさ) を落とさない** — dtype 選択で混乱させない、conventions を強く効かせる
3. **ゼロコスト抽象化** — runtime dispatch を hot path に置かない
4. **汎用言語として健全** — ML 専用に倒さず、科学計算 / 数値解析でも破綻しない
5. **BLAS 統合可能** — Accelerate / OpenBLAS / MKL を将来差し替えられる

## 3. 現状の Almide 型システム (調査結果)

| 機能                          | 実装                                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| ジェネリクス `fn foo[T]`      | ✓ stdlib / user 関数で利用可 (`almide-syntax/src/ast.rs:91`)                             |
| パラメトリック型 `List[T]`   | ✓ `Ty::Applied(TypeConstructorId, Vec<Ty>)` として表現                                   |
| Protocol 制約 `[T: Eq + Ord]` | ✓ 7 built-in (Eq, Repr, Ord, Hash, Codec, Encode, Decode) + ユーザー定義可能             |
| 複数 bounds                   | ✓ `[T: Showable + Nameable]` サポート済み                                                |
| 型推論 (Let-polymorphism)     | ✓ `instantiate_ty` + unification via UnionFind                                           |
| `ConcretizeTypes` パス        | ✓ IR 全体の `IrExpr.ty` を concrete 化                                                   |
| **Primitive の f32/f64 区別** | ✗ **`Ty::Int` = i64, `Ty::Float` = f64 のみ**。f32/f16/i32 等は stdlib 関数名でしか存在しない |
| **`Matrix[T]`**               | ✗ `Ty::Matrix` は非パラメトリック。`Applied(Matrix, [T])` にする必要あり                 |
| モノモフ化                    | Rust 側: rustc に委譲 / WASM 側: runtime dispatch (TypeVar → concrete via pass)          |

**結論**: Matrix[T] 自体は既存の機構の延長で実現可能。
**本当の設計課題は「numeric primitive 型を言語レベルで導入するか」**。

## 4. 他言語の参考事例

| 言語         | アプローチ                             | コスト                    | 汎用性   |
| ------------ | -------------------------------------- | ------------------------- | -------- |
| **Rust**     | 具体型 (f32, f64) + `T: Num` trait bound + mono | ゼロコスト                | 高       |
| **Haskell**  | 型クラス (`Num a => ...`) + dict-passing or specialize | specialize なら速、そうでないとオーバーヘッド | 高       |
| **C++**      | `template<T>`                          | ゼロコスト (mono)         | 高       |
| **Julia**    | multiple dispatch + JIT specialize     | 実質ゼロコスト            | 極高     |
| **Nim**      | parametric + `SomeFloat` concept       | ゼロコスト                | 高       |
| **Go**       | ジェネリクスあるが数値型制約は弱い    | 軽い                      | 中       |
| **Python/NumPy** | runtime dtype tag (実行時ディスパッチ) | dispatch overhead         | 極高 (型安全なし) |
| **PyTorch**  | Tensor の dtype 属性 + dispatch        | dispatch overhead         | 極高 (型安全なし) |
| **MLIR/XLA** | `tensor<MxNxf32>` compile-time type    | ゼロコスト (ML 特化)      | 低 (ML のみ) |
| **Apple MLX** | Python front + Apple Silicon unified memory、lazy graph evaluation | JIT overhead 少、Apple のみ | ML 特化 |

**理想系**: Rust / Julia / Nim 路線。コンパイル時パラメトリック + numeric protocol + モノモフ化。
**避けたい**: NumPy/PyTorch 式の runtime dispatch onlyは hot path で効くと遅い。また型安全性が落ちる。
**意識する対抗**: **MLX** は Apple Silicon 上で NumPy を超えるケースあり (unified memory + lazy graph)。
  Almide の fusion pass は MLX の lazy graph と同じ概念を AOT で実現する位置付け。

## 5. 提案設計

### 5.1 コア: `Matrix[T]` with `T: Numeric` protocol

```almide
// 言語に追加する numeric primitive 型
type f32
type f64
type f16
type i8, i16, i32, i64
type u8, u16, u32, u64

// Numeric protocol (既存の Protocol system で表現)
protocol Numeric {
  fn zero() -> Self
  fn one() -> Self
  fn add(a: Self, b: Self) -> Self
  fn mul(a: Self, b: Self) -> Self
  // ... 必要な演算
}

// これら全ての primitive が Numeric を実装する (builtin impl)

// Matrix 定義: Ty::Applied(Matrix, [T])
type Matrix[T: Numeric] = ...

// stdlib は generic signature を持つ
matrix.ones[T: Numeric](rows: Int, cols: Int) -> Matrix[T]
matrix.mul[T: Numeric](a: Matrix[T], b: Matrix[T]) -> Matrix[T]
matrix.scale[T: Numeric](m: Matrix[T], s: T) -> Matrix[T]
```

**型推論**: `let a: Matrix[f32] = matrix.ones(3, 3)` のように annotation から T を推論。
もしくは `matrix.ones[f32](3, 3)` で明示。

### 5.2 既定 dtype と migration

**既存コード** は暗黙に `f64` を使っている (Matrix = Matrix[f64])。破壊的変更を避けるため:

- `Matrix` を `Matrix[f64]` のエイリアスとして残す (Deprecation warning 付き)
- 新規コードは `Matrix[T]` を明示推奨
- 移行期間: 2 リリース (e.g. 0.14 で warning、0.16 で removal)

### 5.3 エスケープハッチ: `DynMatrix`

IO 境界 (GGUF/GGML ファイル読み込み等) では、ファイルの dtype が実行時まで決まらない。
ここで型パラメトリックを無理に通そうとすると MSR が壊れる。

```almide
// IO 側: dtype が実行時に決まるケース
type DynMatrix = { dtype: Dtype, shape: List[Int], bytes: Bytes }

// 読み込み: 実行時 dtype タグ付き
matrix.load_gguf(path: String) -> Result[DynMatrix, IOError]

// narrowing: 明示的 assertion で静的型に変換
matrix.as[T: Numeric](m: DynMatrix) -> Result[Matrix[T], DtypeError]

// 使用例
let dyn = matrix.load_gguf("whisper.gguf")!
let m: Matrix[f32] = matrix.as[f32](dyn)!  // 実行時に f32 チェック
```

**設計原則**:
- DynMatrix は **IO 境界だけ**で使う (hot path には置かない)
- 一度 narrowing したら以降は静的に `Matrix[T]` として扱う
- `matrix.mul(dyn_a, dyn_b)` のような dispatch API は提供しない
  (NumPy/PyTorch の罠を避ける — 静的に書けたはずのコードが runtime error で落ちる)

**⚠️ 未解決の設計課題**: `Dtype` 型 (type を値として扱う) は **現状 Almide に存在しない**。
以下のサブ設計が必要:
- `Dtype` を `enum Dtype { F32, F64, F16, I32, ... }` として導入
- `matrix.as[T]` は compile-time type parameter T から runtime `Dtype` 値への mapping が必要 (type reflection)
- Rust の `TypeId::of::<T>()` 相当の機構
- **これは DynMatrix (Phase 5) の前提条件**。別 spec `almide/docs/specs/type-reflection.md` で議論予定

## 6. 統合ロードマップ (機能 × 性能)

機能有効化と性能到達を 1 本の Phase 列に統合。各 Phase は「完了条件 (機能)」と「対 NumPy 到達点 (性能)」の両方を持つ。

### Phase 概要表

| Phase | タイトル | 変更範囲 | 言語変更 | 対 NumPy (1024² matmul) | 撤退条件 |
|---|---|---|---|---|---|
| **P0** | 現状 (burn + Accelerate f64 dgemm) | — | — | **1/5** (224 GFLOPS vs 1136) | — |
| **P1** *(optional)* | burn overhead 削減 | runtime のみ | なし | **~1/3.5** (~310 GFLOPS, f64 天井) | ROI 薄ければ skip 推奨 |
| **P2** | Numeric primitive 型 | 型システム | あり (f32,f64,... 型追加) | 変化なし | MSR が 5pt 以上低下したら延期 |
| **P3** | `Numeric` protocol | 型システム | あり (protocol 追加) | 変化なし | - |
| **P4** | `Matrix[T]` パラメトリック化 | 型 + codegen + stdlib | あり | 変化なし (P5 の準備) | 既存コード互換崩壊時は別ブランチ |
| **P5** | BLAS f32 dispatch (P4 依存) | runtime + stdlib | なし | **≈1倍** (タイ, ~1100-1500 GFLOPS) | - |
| **P6** | `DynMatrix` + `Dtype` reflection | 型システム + IO stdlib | あり | 変化なし | reflection 機構が複雑すぎる場合は DynMatrix 取り下げ、Matrix[T] + 個別ローダー関数に |
| **P7** | Shape specialization + in-place rewrite | 新規 2 nanopass | なし | **1.5-3倍** (小行列で圧勝) | ベンチで +20% 未満なら一時凍結 |
| **P8** | Fusion pass | 新規 nanopass | なし | **2-5倍** (推論 E2E) | Whisper E2E で +30% 未満なら fusion patterns 縮小 |
| **P9** | Auto-parallel + WASM SIMD128 | codegen | なし | **NumPy 不可領域** (edge/並列) | - |

**本命ライン**: **P2 → P3 → P4 → P5** で NumPy タイまで到達。P1 は f64 天井 (350 GFLOPS = NumPy の 1/4) までしか行けないので **skip 推奨**。

### Phase 詳細

#### P1 *(optional, skip 推奨)*: burn overhead 削減

**目的**: 現状 236 GFLOPS を f64 天井 ~350 GFLOPS に寄せる。
**ROI**: +50% だが、f64 天井 = NumPy の 1/4 止まりなので投資回収が薄い。
**撤回理由**: 当初「BLAS 統合で 2 倍」と想定したが、調査で burn が既に Accelerate を呼んでいることが判明。
  真の課題は f64 vs f32 の精度差で、これは Matrix[T] (P4) 以降でしか解決できない。

もし実施するなら:
- `matrix_burn.rs:127` の `a.clone().matmul(b.clone())` で cloning overhead を削減
- 入力 Tensor の `from_data` 再構築を回避する経路を探す
- `to_vec()` roundtrip を排除
- **検証**: 1024² で 300+ GFLOPS、Whisper E2E pass

**判断**: ここを飛ばして P2 に進む方が NumPy タイ到達の総時間が短い。

#### P2: Numeric primitive types

- `Ty::F32`, `Ty::F64`, `Ty::F16`, `Ty::I8..U64` を `Ty` enum に追加
- Parser: `f32`, `f64`, `f16` 等を型名として認識
- 既存の `Int` / `Float` は `i64` / `f64` のエイリアスとして維持
- リテラル接尾辞: `1.0_f32`, `42_i32` 等 (Rust 流)
- 推論: 数値リテラルはデフォルト `i64` / `f64`、annotation で narrowing
- **Dojo MSR ベンチ必須**: 新リテラル導入で既存タスクの生存率が変化しないか確認

#### P3: `Numeric` protocol

- built-in protocol として追加 (既存の Eq/Repr と同格)
- 全 numeric primitive type に builtin impl
- `T: Numeric` を parse/resolve できるよう registration 拡張

#### P4: `Matrix[T]` パラメトリック化

- `Ty::Matrix` を `Ty::Applied(TypeConstructorId::Matrix, [T])` に移行
- `TypeConstructorId::Matrix` を kind `* -> *` として登録
- stdlib `matrix.toml` を dtype template 化 (`{dtype}` placeholder)
- codegen dispatch を `(module, func, dtype)` triple key に拡張
- 既存 `Matrix` = `Matrix[f64]` エイリアスで後方互換 (2 リリース後に警告、3 リリース後に removal)
- **Migration owner**: compiler team が 1 リリースサイクル内で nn/whisper 含む全消費者更新

#### P5: BLAS f32 dispatch

- P4 完了後に有効化 (dtype 毎の dispatch がここで初めて言語レベルで通る)
- `cblas_sgemm` を f32 matmul に接続、`cblas_dgemm` は f64 用に維持
- WASM 側は SIMD128 で自前実装 or JS 側 BLAS 呼び出し (wasm-bindgen)
- **検証**: 1024² f32 で 1200+ GFLOPS (NumPy の 90% 以上)

#### P6: `DynMatrix` + `Dtype` reflection

- **前提**: `Dtype` 型 (type を値として扱う) の spec が `almide/docs/specs/type-reflection.md` で確定済
- `DynMatrix` 型と `matrix.as[T]` 関数追加
- GGUF/GGML ローダーを `DynMatrix` を返すよう変更
- 受け側で `matrix.as[f32]` narrowing

#### P7: Shape specialization + in-place rewrite

**着眼点**: LLM/ユーザーが自然に書いた `matrix.ones(3, 3)` の shape は compile-time 定数。

```almide
let small = matrix.ones[f32](3, 3)  // shape=(3,3) が compile time に既知
let big = matrix.ones[f32](1024, 1024)
```

- Compile-time shape → 小行列は BLAS call overhead 回避、inline SIMD op に展開
- Compile-time shape → 大行列は BLAS 呼び出し、buffer 先読み確保
- `BorrowInsertion` を matrix に拡張: linear 所有権の matrix は in-place 書き換え

**新規パス**: `MatrixShapeAnalysis` (shape 伝播) + `MatrixInPlaceRewrite` (所有権から in-place 変換)

#### P8: Fusion pass

**中核**: pipe 連鎖を 1 カーネルに融合。JAX/XLA 相当を **AOT nanopass** で実現。

```almide
// ユーザーが書くコード (LLM も同じように書く)
let y = x
  |> matrix.mul(w1)
  |> matrix.add(b1)
  |> matrix.relu
  |> matrix.mul(w2)
  |> matrix.add(b2)

// fusion pass 後の emit されるカーネル (概念)
fused_linear_relu_linear(x, w1, b1, w2, b2) -> y
  // 1 パスでメモリを walk、SIMD + tile + 中間 alloc 0
```

**nanopass 名称案**: `MatrixFusion`
- 入力: IR の pipe 連鎖 (`|>`) + matrix op 列
- 検出: fuse 可能な op (elementwise, GEMM+bias, GEMM+activation, etc.)
- 出力: 単一 `FusedMatrixKernel` IR ノード
- codegen: dtype + shape + op パターンで特殊化 kernel 生成 (Rust/WASM 両対応)

**fuse 可能パターン初期セット**:
- `mul + add` (GEMM + bias)
- `mul + add + relu/gelu/tanh` (linear layer)
- `add + scale` (broadcast ops)
- `softmax_rows + mul` (attention weights × V)
- elementwise chain (map + map + map)

#### P9: Auto-parallel + WASM SIMD

- **Auto-parallel**: fusion パス後、独立なカーネル同士を並列スケジュール (Rayon / worker thread)
- **WASM SIMD128**: WASM codegen で v128 intrinsics 使用、browser 上でも native 近い性能
- **GPU (将来)**: Metal/CUDA codegen — IR レベルの fusion があれば GPU kernel にも展開可能

## 6.5 ベンチマーク計画

各 Phase の達成判定は以下の 3 本柱で行う。

### B1: Matmul 単体スループット (nn/examples/_bench_matmul.almd)

- サイズ: 384², 384×1536×384, 1536×384×1536, 512², 1024²
- dtype: f32 (主), f64 (検証用)
- 比較対象: NumPy 2.2 (Accelerate) + C raw cblas (upper bound)
- 測定: 1024² で GFLOPS、iter-dependent input で CSE hoist 防止
  (naive impl では `while i < iters { a_i = scale(a, 1 + i*ε); mul(a_i, b) }` で LLVM の invariant hoist を阻止)
- **ベースライン値** (2026-04-14 実測):
  - Almide 現状 = 236 GFLOPS
  - C raw f64 dgemm 天井 = ~350 GFLOPS
  - NumPy f32 = 1362 GFLOPS
  - C raw f32 sgemm 天井 = ~1500 GFLOPS
- **達成基準**:
  - P1 (optional): 1024² f64 で 300+ GFLOPS (burn overhead 削減後)
  - P5: 1024² f32 で 1200+ GFLOPS (NumPy タイ)
  - P7: 小行列 (3², 32², 128²) で NumPy を超える
  - P8: Whisper encoder block E2E で NumPy の 2 倍以上

### B2: Whisper tiny E2E 推論時間 (nn/examples/whisper_demo.almd)

- 30 秒の WAV → テキスト
- 比較対象: whisper.cpp (C++ Accelerate)、openai-whisper (PyTorch)
- **達成基準**:
  - P7: 現状比 +50% 高速化
  - P8: 現状比 +100% 高速化 (pipeline fusion 効果)

### B3: Dojo MSR ベンチ (matrix タスク群)

- Dojo に matrix 関連タスク 10 問を追加 (Phase 2 着手前に必須)
- 各 Phase 前後で LLM 生存率を測定
- **達成基準**: 各 Phase で MSR が -5pt 以上低下しないこと

## 6.6 Almide らしさ (全 Phase 共通で死守)

1. **書きやすさ > 性能チューニング** — `|>` で自然に書いたコードが最速。NumPy の `np.einsum` や `out=` に相当する「性能のための歪んだ書き方」を要求しない
2. **型システム一貫性** — `Matrix[T]` は `List[T]` と同格。特別扱いしない
3. **nanopass 粒度** — fusion も shape specialization も「パス 1 つ追加」で収まる設計に保つ。大きくなる場合は分割
4. **multi-target 対等** — native と WASM で同等に最適化。片方だけ速い/遅いは許容しない
5. **MSR 維持** — 新機能追加後も、LLM が書いたコードの修正生存率が下がらないこと (B3 で定期ベンチ)

## 6.7 結論

**P1 は skip 推奨** (f64 天井 = NumPy の 1/4 止まり、burn overhead 削減の ROI 薄い)。
**P2-P5 が本命ライン**: Numeric primitive 型 → Numeric protocol → Matrix[T] → f32 dispatch
で NumPy タイまで到達。**P7-P8 で構造的に超える** (shape spec + fusion)。

Almide の強みは「最速コードが自然に書ける」そのもの。ML 特化ではなく
**AOT ナノパス言語の強みを汎用に活かす**方向で設計する。

## 7. 決定事項 (closed)

- **リテラル dtype 既定**: Rust 流。annotation から推論、デフォルト `f64` / `i64`、`1.0_f32` で narrowing
- **f16/bf16**: Phase 2 では除外。P9 以降で ML 特化拡張として追加
- **既存コードへの破壊性**: デフォルトパラメータ `matrix.ones[T = f64](...)` で後方互換。`Matrix` は `Matrix[f64]` エイリアスとして 2 リリース維持

## 8. 未決事項 (open)

- **Matrix の内部表現**: `Vec<Vec<T>>` → `Vec<T>` + shape の contiguous 化は BLAS 統合で必要だが、P1 では flatten/unflatten で吸収し別 spec (`almide/docs/specs/matrix-layout.md`) で議論
- **Dtype reflection 機構**: DynMatrix の前提となる「型を値として扱う」機構。P6 前に別 spec (`almide/docs/specs/type-reflection.md`) で確定必要
- **リテラル dtype のコンテキスト依存推論**: `let x: f32 = 1.0` の `1.0` は f32 に narrow されるか、default f64 のまま推論失敗か
- **Migration ownership**: P4 で `Matrix` → `Matrix[f64]` 移行時、nn / whisper / ユーザープロジェクトの更新責任範囲

## 9. 参考

- Rust ndarray: `Array<A, D>` where A: numeric
- Julia: `Matrix{Float32}` / `Matrix{Float64}` with multiple dispatch
- Nim arraymancer: `Tensor[T]` with `SomeFloat` bound
- NumPy/PyTorch: runtime dtype tag (反面教師)
- Apple MLX: unified memory + lazy graph evaluation (対抗ターゲット)

## 10. このメモの位置付け

- **nn 側の doc** として置くが、実体は **Almide コンパイラ側の仕様議論**
- 実装時は本メモを `almide/docs/specs/matrix-dtype.md` に昇格させる
- Phase 2 以降の着手前に **Dojo に matrix MSR タスク 10 問**を追加 (B3 前提)
- 関連 sub-spec (未作成):
  - `almide/docs/specs/matrix-layout.md` — Matrix 内部表現 (Vec<Vec<T>> vs contiguous)
  - `almide/docs/specs/type-reflection.md` — Dtype reflection 機構
