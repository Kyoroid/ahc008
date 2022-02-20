# Usage

## Requirements
Please install a compiler for Rust language (see https://www.rust-lang.org).
For those who are not familiar with the Rust language environment, we have prepared a [pre-compiled binary for Windows](https://img.atcoder.jp/ahc008/tools_x86_64-pc-windows-gnu_v3.zip).
The following examples assume that you will be working in the directory where this README is located.

## Input Generation
The `in` directory contains the pre-generated input files for seed=0-99.
If you want more inputs, prepare `seeds.txt` which contains a list of random seeds (unsigned 64bit integers) and execute the following command.
```
cargo run --release --bin gen seeds.txt
```
This will output input files into `in` directory.

If you use the precompiled binary for Windows, replace `cargo run --release --bin gen` with `./gen.exe`.

## Testing
Let `in.txt` be an input file, `out.txt` be a file to which output of your program will be written, and `cmd` be a command to run your program.
You can test your program by executing the following command.
```
cargo run --release --bin tester cmd < in.txt > out.txt
```
The above command outputs the score to standard error.
You can specify arguments to your program by 
```
cargo run --release --bin tester cmd arg1 arg2 ... < in.txt > out.txt
```

If you use the precompiled binary for Windows, replace `cargo run --release --bin tester` with `./tester.exe`.


### Examples
When executing a binary `a.out` which has been compiled in C++, etc. 
```
cargo run --release --bin tester ./a.out < in.txt > out.txt
```

When executing a program `main.py` using python3.
```
cargo run --release --bin tester python3 main.py < in.txt > out.txt
```

### When using Powershell on Windows
Because you cannot use `<` in the above examples, please execute as follows.
```
cat in.txt | cargo run --release --bin tester cmd > out.txt
```

For some execution commands, you may need to specify the full path, as in the following example.
```
cat in.txt | cargo run --release --bin tester C:\Users\myname\AppData\Local\Microsoft\WindowsApps\python3.exe main.py > out.txt
```
You can check the full path of a command by executing `gcm python3`.


## Visualization
Let `in.txt` be an input file and `out.txt` be an output file.
You can visualize the output by executing the following command.
```
cargo run --release --bin vis in.txt out.txt
```
The above command writes a visualization result to `vis.html`.
You can also use a [web visualizer](https://img.atcoder.jp/ahc008/f828b9475ffb41d54f05619db6ccbd4f.html?lang=en) which is more rich in features.
To use the web visualizer, first generate an output file `out.txt` as described in the Testing section, and paste its contents into the Output filed of the visualizer.
By changing the number in the Seed field, you can switch to an input for another seed.
By pressing the `▶` button, animation will start.

## Acknowledgements
We used images from [DOTOWN](https://dotown.maeda-design-room.net/).


# 使い方

## 実行環境
Rust言語のコンパイル環境が必要です。
https://www.rust-lang.org/ja を参考に各自インストールして下さい。
Rust言語の環境構築が面倒な方向けに、[Windows用のコンパイル済みバイナリ](https://img.atcoder.jp/ahc008/tools_x86_64-pc-windows-gnu_v3.zip)も用意してあります。
以下の実行例では、このREADMEが置かれているディレクトリに移動して作業することを想定しています。

## 入力生成
`in` ディレクトリに予め生成された seed=0~99 に対する入力ファイルが置かれています。
より多くの入力が欲しい場合は、`seeds.txt` に欲しい入力ファイルの数だけ乱数seed値(符号なし64bit整数値)を記入し、以下のコマンドを実行します。
```
cargo run --release --bin gen seeds.txt
```
生成された入力ファイルは `in` ディレクトリに出力されます。

Windows用のコンパイル済バイナリを使用する場合は `cargo run --release --bin gen` の部分を `./gen.exe` に置き換えて下さい。

## テスト実行
入力ファイル名を`in.txt`、出力結果を書き出す先のファイル名を`out.txt`、あなたのプログラムの実行コマンドを`cmd`としたとき、以下のコマンドを実行します。
```
cargo run --release --bin tester cmd < in.txt > out.txt
```
実行が終わると、スコアが標準エラーに出力されます。
引数が必要な場合には
```
cargo run --release --bin tester cmd arg1 arg2 ... < in.txt > out.txt
```
のようにします。

Windows用のコンパイル済バイナリを使用する場合は `cargo run --release --bin tester` の部分を `./tester.exe` に置き換えて下さい。

### 例
C++などでコンパイル済みの `a.out` というバイナリを実行する場合
```
cargo run --release --bin tester ./a.out < in.txt > out.txt
```

python3 で `main.py` というプログラムを実行する場合
```
cargo run --release --bin tester python3 main.py < in.txt > out.txt
```

### Windows で Powershell を使用する場合
上記の `<` が使えないため、以下のように実行して下さい。
```
cat in.txt | cargo run --release --bin tester cmd > out.txt
```

実行コマンドによっては以下の例のようにフルパスで指定する必要がある場合があります。
```
cat in.txt | cargo run --release --bin tester C:\Users\myname\AppData\Local\Microsoft\WindowsApps\python3.exe main.py > out.txt
```
コマンドのフルパスは `gcm python3` というコマンドを実行することで確認出来ます。

## ビジュアライザ
入力ファイル名を`in.txt`、出力ファイル名を`out.txt`としたとき、以下のコマンドを実行します。
```
cargo run --release --bin vis in.txt out.txt
```
出力のビジュアライズ結果は `vis.html` というファイルに書き出されます。
より機能が豊富な[ウェブ版のビジュアライザ](https://img.atcoder.jp/ahc008/f828b9475ffb41d54f05619db6ccbd4f.html?lang=ja)も利用可能です。
ウェブ版のビジュアライザを使用するには、まずテスト実行の項目を参考に出力ファイル `out.txt` を作成し、その中身をビジュアライザのOutput欄に貼り付けて下さい。
Seed欄の数字を変更することで他のseedの入力に切り替わります。
`▶` ボタンを押すとアニメーションが開始します。

## 謝辞
画像は [DOTOWN](https://dotown.maeda-design-room.net/) のものを利用させていただきました。
