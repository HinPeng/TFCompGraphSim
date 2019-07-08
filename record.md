||resnet50||
|-----|--------|-----|
|no optimize|252.04(190)|
|on demand|55.88(650) memory:57G|
|no updateswapinfo|60|
|undateswapinfo|184.15(236)|22/45(not swap in when recompute)|

# MM

## resnet50
recompute depth max is 2
max batch size with recompute policy depth=2

## bert
    1. recompute hang out
        - ['attention/output/add:0', 'attention/self/Reshape_3:0', 'attention/self/dropout/mul:0', 'attention/self/Softmax:0']
        - example: bert/encoder/layer_11/attention/output/add:0
    2. couldn't find tensor
        -  'bert/embeddings/one_hot', 'bert/embeddings/LayerNorm/batchnorm/Rsqrt', 'attention/output/LayerNorm/batchnorm/add_1'

    3. 
