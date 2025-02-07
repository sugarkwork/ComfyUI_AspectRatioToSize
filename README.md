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

