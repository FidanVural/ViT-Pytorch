# ViT-Pytorch
Bu repoda Vision Transformer mimarisinin Pytorch kütühanesi ile implementasyonu bulunmaktadır.

### Adım Adım Vision Transformer
İlk olarak aşağıdaki görselde mimarinin tamamı gösterilmektedir. Genel anlamda özetlersek ilk olarak görüntü alınarak belirlenen sayıda patch'e ayrılır. Elde edilen patch'ler vektörlere çevrilerek bir dizi elde edilir ve bu dizinin en başına öğrenilebilir bir paramtre olan classification token eklenir. Encoder'a verilmeden önce son olarak bu diziye konum bilgisi eklenir (positional encoding). Bu dizi Transformer Encoder'a girer ve MLP(Multi Layer Perceptron) kısmında da classification token ile sınıflandırma işlemi gerçekleştirilir.

<p align="center">
  <img src="https://user-images.githubusercontent.com/56233156/236134751-56bcbcc0-6b6b-48fe-a283-aefac390da0f.png">
</p>
<p align="center"> 
    <em>Vision Transformer Model Mimarisi [1]</em>
</p>

#### 1) Splitting an Image Into Patches and Linear Mapping

Bu kısımda görüntü belli bir patch (parça) sayısına göre parçalara ayrılır. NxNxC'lik bir görüntü için patch boyutunu PxP olarak belirlediğimizde elde edilen toplam patch sayısı (H\*W\*C)/(P\*P) olur. Örneğin elimizde 28x28x1 boyutunda bir görsel olsun. Eğer patch boyutunu 7x7 olarak belirlersek elde edeceğimiz toplam patch sayısı 16 olur ve oluşan her bir patch boyutu da 4x4x1 olur.
Sonrasında 2D olan her bir patch 1D haline getirilir. Bu da P\*P\*C işlemi ile gerçekleştirilir. Önceki örneğimizi düşündüğümüzde vektörün uzunluğu 4x4x1'den 16 olur.
Elde edilen vektör linear mapping ile istenilen boyuta eşlenebilir. Kodda hidden_d ile belirtilen değer aslında dizinin embedding size olarak kaça eşleneceğini belirtir.

<p align="center">
  <img src="https://user-images.githubusercontent.com/56233156/236189732-a961bbf3-9860-4500-a57a-2c9ef21f7dc8.png">
</p>

#### 2) Adding CLassification Token
Classification token elde edilen dizinin başına eklenir. Bu eklenen token değeri öğrenilebilir bir parametredir.

#### 3) Positional Encoding
Positional encoding ile de elde edilen bu patch embedding dizisine konum bilgisi eklenir. Transformer yapıları girdilerin sırasını hatırlama yeteneğine sahip değildir. Bu nedenle görüntüdeki patch'lerin sırasının değişmesi demek görüntünün anlamının kaybolması demektir. Bu nedenle positional encoding'e ihtiyaç duyarız. 
Positional encoding için kullanılan formüller aşağıdaki görselde verilmiştir. O satırdaki çift konumlu değerler için sin fonskiyonu tek satırlı değerler için cos fonskiyonu kullanılmıştır.

<p align="center">
  <img src="https://user-images.githubusercontent.com/56233156/236178408-2e9c5704-4445-4cea-97b6-2d2963265304.png">
</p>


#### 4) TRANSFORMER ENCODER

##### 4.1) Multi-Head Attention



