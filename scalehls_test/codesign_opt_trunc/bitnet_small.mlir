#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map7 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
module attributes {torch.debug_module_name = "MinimalAttention"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x32x64xf32>) -> tensor<1x32x32xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<"0xB2A0C53D0679EDBD9C0429BDA8738C3D1672AABD9E42A83D240225BD1C8AA53DA045743C20E41BBC76E5D6BDA0816FBC8C1E28BD0E22C9BD0A06E53DD0EC08BDC06CE13CD08A5B3C842513BD9015F0BCD8AD58BD02F8D7BDBE10F3BD6C56B0BDF496D13DB033BABCE8DB033DC4D50BBD5EAAE63D50A96BBDB015C2BC00E42D3DF4C0563DC0C51A3BE04283BC0E1693BD904E42BD9C8FF03D185DAEBC0CA6813D1493673DE05F413D3083563C629DFD3D40D3893CF0716D3D1A2FC33D40F803BDC0A4B8BB5ACBFBBD50C3663D48CBF1BD9C43B63D8080CF3DB46C4DBDD09BAFBDC2678E3D00ECAB3A8081503BA45554BD162EE7BD0068D83CF8EAB7BDC849DC3CB0A4CABDCC22B03DE4B51BBD0A5DC93DC6B9C6BDC0116D3B16A3983D6403F13D10493A3C84FEB2BD00DB4F3DB2B6F93D3C0C38BDC84DB0BC404AC63D2092A8BCD2F99FBD8840BB3D80E9FF3B9832EF3C3C1B9EBD9C345ABD1030A03DA0484EBDCC0DA93D00F42D39C0994A3CDC974EBDE06A453C34AD0ABD30BA5A3D007DF33C302A293CB2F0963DB03155BCE486B5BDC6D7A7BD3A55C53DC865F0BCE207CFBD3A3CE3BD20ECC63C28F5DCBD8618843DA8B33FBD103C45BDD8596A3D1C14D93DC8FCA23CF056EABCB235BCBD56DBCBBDFAE5F13DF8C75F3DFCB4353D5005023C1C4B
    %cst_0 = arith.constant dense<"0x2E739CBD08F4D53C2EBDB0BDC04A803CF890773DF064A13D6841D4BDD4DE0D3D70EAE23DB0ED413C9252B9BDA857473D70B279BC809BD13B4E5ED3BD2C46BEBD6A509F3D482D0F3D9003E13DF49C773D203D743DF03AD03CF4F0D73DD4751F3D2290D33D26DC823D2C53F5BDFA0AA1BD305AB2BC401C833CFC07D6BD0A2CF6BD189804BDD82C91BCC052753DA0B787BC2439B6BDE241FCBDCCE48D3D34F8483D3015033C5EBBD83D34DAC13D0434193D00EEBCBA2EEADDBD90845C3CD698DD3DC0D3753B2CF8CB3D72F5D23DD02AF1BC6051B4BDC84B99BC78351A3DE6EFB53D8AC29D3DF64EA8BD10DBE7BDCAB5CFBD0AC39B3D5A0FA23D8C7775BDFE7CBD3D86FCE33D16C2A33DFA8FE3BDAAA3C7BD50A8EDBD6EE6F43D72D9D1BD00445D3BD804AA3C4E30D1BD4C47F53DA0004C3DDEB8BDBD50B5ACBD1AA8A4BD042F2ABDBE75DE3D7047EC3DA046A7BBF61BFFBDC896E93DA805BD3C2473D4BD00E2F4BD4455EA3D7C70FF3D6E5FBCBDDC90573DDC1BDF3DDCF43BBD20E4F63B58AA6FBDDCAE7F3DAEE88C3DB028703DF4A9AFBD80128B3D44A53BBDCC35DF3DB411C1BD621FD0BD68DFCC3C70ADBFBC6076B13DF277A23D90E89FBD268A8FBD507146BDE0B1CC3DE867CD3C50550CBD50239ABCE8BCA23D2400E0BD00C31C3CDC2A903DD0
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    %2 = tensor.empty() : tensor<1x32x64xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x32x64xf32>) outs(%2 : tensor<1x32x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x64xf32>
    %4 = tensor.empty() : tensor<1x64x64xf32>
    %5 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<64x64xf32>) outs(%4 : tensor<1x64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x64xf32>
    %6 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<1x32x64xf32>) -> tensor<1x32x64xf32>
    %7 = linalg.batch_matmul ins(%3, %5 : tensor<1x32x64xf32>, tensor<1x64x64xf32>) outs(%6 : tensor<1x32x64xf32>) -> tensor<1x32x64xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    %9 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<64x64xf32>) outs(%4 : tensor<1x64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x64xf32>
    %10 = linalg.batch_matmul ins(%3, %9 : tensor<1x32x64xf32>, tensor<1x64x64xf32>) outs(%6 : tensor<1x32x64xf32>) -> tensor<1x32x64xf32>
    %11 = tensor.empty() : tensor<1x64x32xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<1x32x64xf32>) outs(%11 : tensor<1x64x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x64x32xf32>
    %13 = tensor.empty() : tensor<1x32x32xf32>
    %14 = linalg.fill ins(%cst_1 : f32) outs(%13 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
    %15 = linalg.batch_matmul ins(%7, %12 : tensor<1x32x64xf32>, tensor<1x64x32xf32>) outs(%14 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>
    %16 = tensor.empty() : tensor<1x32x1xi64>
    %17 = linalg.fill ins(%c0_i64 : i64) outs(%16 : tensor<1x32x1xi64>) -> tensor<1x32x1xi64>
    %18 = tensor.empty() : tensor<1x32x1xf32>
    %19 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %20:2 = linalg.generic {indexing_maps = [#map3, #map6, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : tensor<1x32x32xf32>) outs(%19, %17 : tensor<1x32x1xf32>, tensor<1x32x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_3: i64):
      %26 = linalg.index 2 : index
      %27 = arith.index_cast %26 : index to i64
      %28 = arith.maxf %in, %out : f32
      %29 = arith.cmpf ogt, %in, %out : f32
      %30 = arith.select %29, %27, %out_3 : i64
      linalg.yield %28, %30 : f32, i64
    } -> (tensor<1x32x1xf32>, tensor<1x32x1xi64>)
    %21 = linalg.generic {indexing_maps = [#map2, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %20#0 : tensor<1x32x32xf32>, tensor<1x32x1xf32>) outs(%13 : tensor<1x32x32xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %26 = arith.subf %in, %in_3 : f32
      linalg.yield %26 : f32
    } -> tensor<1x32x32xf32>
    %22 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21 : tensor<1x32x32xf32>) outs(%13 : tensor<1x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = math.exp %in : f32
      linalg.yield %26 : f32
    } -> tensor<1x32x32xf32>
    %23 = linalg.fill ins(%cst_1 : f32) outs(%18 : tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %24 = linalg.generic {indexing_maps = [#map3, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%22 : tensor<1x32x32xf32>) outs(%23 : tensor<1x32x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %26 = arith.addf %in, %out : f32
      linalg.yield %26 : f32
    } -> tensor<1x32x1xf32>
    %25 = linalg.generic {indexing_maps = [#map2, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22, %24 : tensor<1x32x32xf32>, tensor<1x32x1xf32>) outs(%13 : tensor<1x32x32xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %26 = arith.divf %in, %in_3 : f32
      linalg.yield %26 : f32
    } -> tensor<1x32x32xf32>
    return %25 : tensor<1x32x32xf32>
  }
}

