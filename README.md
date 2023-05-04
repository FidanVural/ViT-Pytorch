# ViT-Pytorch
Bu repoda Vision Transformer mimarisinin Pytorch kütühanesi ile implementasyonu bulunmaktadır.

### Adım Adım Vision Transformer
İlk olarak aşağıdaki görselde mimarinin tamamı gösterilmektedir. Genel anlamda özetlersek ilk olarak görüntü alınarak belirlenen sayıda patch'e ayrılır. Elde edilen patch'ler vektörlere çevrilerek bir dizi elde edilir ve bu dizinin en başına öğrenilebilir bir parametre olan classification token eklenir. Encoder'a verilmeden önce son olarak bu diziye konum bilgisi eklenir (positional encoding). Bu dizi Transformer Encoder'a girer ve MLP(Multi Layer Perceptron) kısmında da classification token ile sınıflandırma işlemi gerçekleştirilir.

<p align="center">
  <img src="https://user-images.githubusercontent.com/56233156/236134751-56bcbcc0-6b6b-48fe-a283-aefac390da0f.png">
</p>
<p align="center"> 
    <em>Vision Transformer Model Mimarisi [1]</em>
</p>

#### 1) Splitting an Image Into Patches and Linear Mapping

Bu kısımda görüntü belli bir patch (parça) sayısına göre parçalara ayrılır. NxNxC'lik bir görüntü için patch boyutunu PxP olarak belirlediğimizde elde edilen toplam patch sayısı (H\*W\*C)/(P\*P) olur. Örneğin elimizde 28x28x1 boyutunda bir görsel olsun. Eğer patch boyutunu 7x7 olarak belirlersek elde edeceğimiz toplam patch sayısı 16 olur ve oluşan her bir patch boyutu da 4x4x1 olur. Daha iyi anlamak için aşağıdaki şekildeki görselleştirmeye bakılabilir. Ayrıca patch'lere ayırma ve görselleştirme için ... kodu da incelenebilir.

Sonrasında 2D olan her bir patch 1D haline getirilir. Bu da P\*P\*C işlemi ile gerçekleştirilir. Önceki örneğimizi düşündüğümüzde vektörün uzunluğu 4x4x1'den 16 olur. Elde edilen vektör linear mapping ile istenilen boyuta eşlenebilir. Kodda hidden_d ile belirtilen değer aslında dizinin embedding size olarak kaça eşleneceğini belirtir.

<p align="center">
  <img src="https://user-images.githubusercontent.com/56233156/236192461-375232e6-8abb-46da-9445-b901cf12b255.png">
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

<p align="center">
  <img src="https://user-images.githubusercontent.com/56233156/236195906-9aa903ae-1cda-4a0e-b550-2fda8d7c4ee6.png">
</p>

##### 4.1) Multi-Head Attention
Transformer mimarisinin en önemli kısmı diyebiliriz. Burada Query(Q), Key(K) ve Value(V) olarak adlandırılan 3 adet matristen bahsedeceğiz. Her bir 


#### Sources
- https://arxiv.org/pdf/2010.11929.pdf
- https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
- https://medium.com/swlh/visual-transformers-a-new-computer-vision-paradigm-aa78c2a2ccf2
- https://medium.com/machine-intelligence-and-deep-learning-lab/vit-vision-transformer-cc56c8071a20
- https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853#:~:text=This%20logical%20split%20is%20done%20by%20partitioning%20the%20input%20data%20as%20well%20as%20the%20Linear%20layer%20weights%20uniformly%20across%20the%20Attention%20heads.%20We%20can%20achieve%20this%20by%20choosing%20the%20Query%20Size%20as%20below%3A
- https://mertcobanov.medium.com/vision-transformers-nedir-vits-nedir-14dce4d1c6d7
