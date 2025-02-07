# ComfyUI_AspectRatioToSize

アスペクト比を維持した、横幅と高さの数値を計算する、ComfyUI 用のノードです。

任意のアスペクト比を維持した、複数の解像度の計算をする際に使用します。

## AspectRatioToSize
アスペクト比を計算して、サイズオブジェクトを返します。

### input
- aspect_ratio: アスペクト比を指定します。
- resolution: 長辺サイズを指定します。

### output
- Size: Size オブジェクトを返します。


## SizeToWidthHeight
サイズオブジェクトから横幅と高さの値を取り出します。

### input
- Size: サイズオブジェクトを指定します。

### output
- Width: 幅を返します
- Height: 高さを返します
- LongerSide: 長辺のサイズを返します
- SmallSide: 短辺のサイズを返します

# メリット

例えば、16:9 で、長辺が 1920 の解像度を計算して、それを Empty Latent に入力したい、と考えた時、自動計算しようとすると面倒でした。

このノードに、縦横比と長辺の値を指定するだけで、自動的に短辺のサイズも決まり、欲しい解像度の画像サイズの計算が楽になります。
