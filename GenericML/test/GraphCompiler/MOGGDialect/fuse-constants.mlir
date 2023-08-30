// RUN: mo-opt --mo-to-mogg --fuse-elementwise --canonicalize --cse --fuse-bcasts --canonicalize --cse --fuse-consts %s | FileCheck %s

mo.graph @basic_constant_fuse(%arg0: !mo.tensor<[2, 2, 4], f32>) -> (!mo.tensor<[2, 2, 4], f32>) {
  %cst1 = mo.constant {value = #M.dense_array<1.500000e-01, 1.500000e-01, 2.500000e-01, 3.500000e-01, 4.500000e-01, 5.500000e-01, 6.500000e-01, 7.500000e-01, 1.500000e-01, 1.500000e-01, 2.500000e-01, 3.500000e-01, 4.500000e-01, 5.500000e-01, 6.500000e-01, 7.500000e-01> : tensor<16xf32>} : !mo.tensor<[2, 2, 4], f32>
  %cst2 = mo.constant {value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00> : tensor<16xf32>} : !mo.tensor<[2, 2, 4], f32>
  %cst3 = mo.constant {value = #M.dense_array<15.000000e+00, 14.000000e+00, 13.000000e+00, 12.000000e+00, 11.000000e+00, 10.000000e+00, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00, 5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00, 0.000000e+00> : tensor<16xf32>} : !mo.tensor<[2, 2, 4], f32>

  %1 = mo.add(%arg0, %cst1) : !mo.tensor<[2, 2, 4], f32>
  %2 = mo.sub(%1, %cst2) : !mo.tensor<[2, 2, 4], f32>
  %3 = mo.mul(%2, %cst3) : !mo.tensor<[2, 2, 4], f32>
  mo.output %3 : !mo.tensor<[2, 2, 4], f32>
}

// CHECK-LABEL: mo.graph @basic_constant_fuse
// CHECK: (%[[ARG:.*]]: !mo.tensor<[2, 2, 4], f32>)

// Check kernel is next (so consts have been dce'd) & that it has one arg.
// CHECK-NEXT: mogg.kernel (%[[ARG]])
// CHECK-NEXT: mogg.output_placeholders

// Check the constants are inside the kernel now.
// CHECK-NEXT: mo.constant {value = #M.dense_array<1.500000e+01,
// CHECK-NEXT: mo.constant {value = #M.dense_array<1.000000e+00,
// CHECK-NEXT: mo.constant {value = #M.dense_array<1.500000e-01


// Check broadcasted constants get fused
mo.graph @bcasted_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], f32>) -> !mo.tensor<[D14, D15], f32> {
  %0 = mo.constant {value = #M.dense_array<1.250000e-01> : tensor<f32>} : !mo.tensor<[], f32>
  %1 = mo.constant {value = #M.dense_array<6.000000e+02> : tensor<f32>} : !mo.tensor<[], f32>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], f32> -> !mo.tensor<[D14, D15], f32>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], f32> -> !mo.tensor<[D14, D15], f32>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], f32>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], f32>
  mo.output %5 : !mo.tensor<[D14, D15], f32>
}


// CHECK-LABEL: mo.graph @bcasted_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1.250000e-01>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6.000000e+02>
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]

// Check broadcasted f64 constants get fused
mo.graph @bcasted_f64_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], f64>) -> !mo.tensor<[D14, D15], f64> {
  %0 = mo.constant {value = #M.dense_array<1.250000e-01> : tensor<f64>} : !mo.tensor<[], f64>
  %1 = mo.constant {value = #M.dense_array<6.000000e+02> : tensor<f64>} : !mo.tensor<[], f64>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], f64> -> !mo.tensor<[D14, D15], f64>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], f64> -> !mo.tensor<[D14, D15], f64>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], f64>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], f64>
  mo.output %5 : !mo.tensor<[D14, D15], f64>
}


// CHECK-LABEL: mo.graph @bcasted_f64_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1.250000e-01>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6.000000e+02>
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]

// Check broadcasted si32 constants get fused
mo.graph @bcasted_si32_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], si32>) -> !mo.tensor<[D14, D15], si32> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
  %1 = mo.constant {value = #M.dense_array<6> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], si32> -> !mo.tensor<[D14, D15], si32>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], si32> -> !mo.tensor<[D14, D15], si32>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], si32>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], si32>
  mo.output %5 : !mo.tensor<[D14, D15], si32>
}

// CHECK-LABEL: mo.graph @bcasted_si32_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6>
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]


// Check broadcasted si64 constants get fused
mo.graph @bcasted_si64_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], si64>) -> !mo.tensor<[D14, D15], si64> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %1 = mo.constant {value = #M.dense_array<6> : tensor<si64>} : !mo.tensor<[], si64>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], si64> -> !mo.tensor<[D14, D15], si64>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], si64> -> !mo.tensor<[D14, D15], si64>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], si64>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], si64>
  mo.output %5 : !mo.tensor<[D14, D15], si64>
}

// CHECK-LABEL: mo.graph @bcasted_si64_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6>
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]


// Check broadcasted si16 constants get fused
mo.graph @bcasted_si16_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], si16>) -> !mo.tensor<[D14, D15], si16> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<si16>} : !mo.tensor<[], si16>
  %1 = mo.constant {value = #M.dense_array<6> : tensor<si16>} : !mo.tensor<[], si16>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], si16> -> !mo.tensor<[D14, D15], si16>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], si16> -> !mo.tensor<[D14, D15], si16>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], si16>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], si16>
  mo.output %5 : !mo.tensor<[D14, D15], si16>
}

// CHECK-LABEL: mo.graph @bcasted_si16_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6>
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]


// Check broadcasted si8 constants get fused
mo.graph @bcasted_si8_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], si8>) -> !mo.tensor<[D14, D15], si8> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<si8>} : !mo.tensor<[], si8>
  %1 = mo.constant {value = #M.dense_array<6> : tensor<si8>} : !mo.tensor<[], si8>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], si8> -> !mo.tensor<[D14, D15], si8>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], si8> -> !mo.tensor<[D14, D15], si8>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], si8>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], si8>
  mo.output %5 : !mo.tensor<[D14, D15], si8>
}

// CHECK-LABEL: mo.graph @bcasted_si8_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6>
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]


// Check broadcasted ui8 constants get fused
mo.graph @bcasted_ui8_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], ui8>) -> !mo.tensor<[D14, D15], ui8> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<ui8>} : !mo.tensor<[], ui8>
  %1 = mo.constant {value = #M.dense_array<6> : tensor<ui8>} : !mo.tensor<[], ui8>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], ui8> -> !mo.tensor<[D14, D15], ui8>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], ui8> -> !mo.tensor<[D14, D15], ui8>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], ui8>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], ui8>
  mo.output %5 : !mo.tensor<[D14, D15], ui8>
}

// CHECK-LABEL: mo.graph @bcasted_ui8_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6>
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]


// Check broadcasted ui16 constants get fused
mo.graph @bcasted_ui16_const_fusion<D14, D15>(%arg0: !mo.tensor<[D14, D15], ui16>) -> !mo.tensor<[D14, D15], ui16> {
  %0 = mo.constant {value = #M.dense_array<1> : tensor<ui16>} : !mo.tensor<[], ui16>
  %1 = mo.constant {value = #M.dense_array<6> : tensor<ui16>} : !mo.tensor<[], ui16>
  %2 = mo.static.broadcast_to(%0) : !mo.tensor<[], ui16> -> !mo.tensor<[D14, D15], ui16>
  %3 = mo.static.broadcast_to(%1) : !mo.tensor<[], ui16> -> !mo.tensor<[D14, D15], ui16>
  %4 = mo.add(%2, %arg0) : !mo.tensor<[D14, D15], ui16>
  %5 = mo.div(%4, %3) : !mo.tensor<[D14, D15], ui16>
  mo.output %5 : !mo.tensor<[D14, D15], ui16>
}

// CHECK-LABEL: mo.graph @bcasted_ui16_const_fusion
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check constants are in kernel and used in broadcasts.
// CHECK-NEXT: %[[CST1:.*]] = mo.constant {value = #M.dense_array<1>
// CHECK-NEXT: %[[CST2:.*]] = mo.constant {value = #M.dense_array<6>
// CHECK-NEXT: mogg.param_shape_to_value
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST2]]
// CHECK-NEXT: mogg.call "mo.static.broadcast_to"(%[[CST1]]


// Check transpose permutation gets fused
mo.graph @transpose_fold_perm<D14, D15, D16>(%arg0: !mo.tensor<[D14, D15, D16], f32>) -> !mo.tensor<[D16, D15, D14], f32> {
  %perm = mo.constant {value = #M.dense_array<2, 1, 0> : !M.array<3xsi64>} : !mo.tensor<[3], si64>
  %trans = mo.transpose(%arg0, %perm) : (!mo.tensor<[D14, D15, D16], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D16, D15, D14], f32>
  mo.output %trans : !mo.tensor<[D16, D15, D14], f32>
}

// CHECK-LABEL: mo.graph @transpose_fold_perm
// CHECK-NEXT: mogg.kernel
// CHECK-NEXT: mogg.output_placeholders

// Check permutation is in kernel and used in transpose.
// CHECK-NEXT: %[[PERM:.*]] = mo.constant {value = #M.dense_array<2, 1, 0> 
// CHECK-NEXT: mogg.call "mo.transpose"(%[[NULL:.*]], %[[PERM]]

mo.graph @fuse_dense_resource(%input: !mo.tensor<[12, 64], f32>) -> !mo.tensor<[12, 64], f32> {
  %cst = mo.constant {value = dense_resource<const_6> : tensor<12x64xf32>} : !mo.tensor<[12, 64], f32>
  %84 = mo.add(%cst, %input) : !mo.tensor<[12, 64], f32>
  mo.output %84 : !mo.tensor<[12, 64], f32>
}

// CHECK-LABEL: mo.graph @fuse_dense_resource
// CHECK: mogg.kernel
// CHECK: mo.constant

mo.graph @fuse_rebind(%input: !mo.tensor<[2, 2], si32>) -> (!mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], si32>) {
  %cst = mo.constant {value = #M.dense_array<0, 1, 2, 3> : !M.array<4xsi32>} : !mo.tensor<[2, 2], si32, {layout = #mo.layout<KN>}>
  %cst_reb = mo.rebind(%cst) : !mo.tensor<[2, 2], si32, {layout = #mo.layout<KN>}> -> !mo.tensor<[2, 2], si32, {layout = #mo.layout<NK>}>
  %mm1 = mo.matmul(%input, %cst) : (!mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], si32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[2, 2], si32>
  %mm2 = mo.matmul(%input, %cst_reb) : (!mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], si32, {layout = #mo.layout<NK>}>) -> !mo.tensor<[2, 2], si32>
  mo.output %mm1, %mm2 : !mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], si32>
}

// Check constants are in kernels
// CHECK-LABEL: mo.graph @fuse_rebind
// CHECK: mogg.kernel
// CHECK: mo.constant
// CHECK: mogg.output
// CHECK: mogg.kernel
// CHECK: mo.constant
// CHECK: mogg.output

{-#
  dialect_resources: {
    builtin: {
      const_6: "0x0400000023D889BEB425BC3D3D236BBE5426A73D2E00763E405F35BC2AA400BE882D80BEC67A7E3E2092873DB61401BEDEA4643E3CBEEFBD486CB8BD457E833E4CFD8BBD7077733D604406BCFC9B8ABE00424FBEE0458ABC529069BE2C78D7BDD056273D2E6003BEA9511EBED40FCF3DD24CDEBD6C4189BDE254323EA0CB9BBD72FC8BBE80CEDDBBA02BC83C804A1A3B382E6BBD8C73193ECC17E93D7637703E3C15B33DA22AEBBDDE46C7BD0219ECBDDCD22C3EA8CF203DFAF9633E19A953BE60FB48BCB029EC3D38E929BD0E13ABBDD258453E5016D6BC7B3461BEF00BDD3C8697693E14CF49BD63478D3E1A9723BEB0DE8E3DB6DA6E3ED6362E3E3011403E08831C3EB8D8513E2EC1733E0F32843EF05FFE3C30BE9BBC1F4189BECB7A873E8245683EC8596D3DE38919BEF4F9F23DD816AABD3C4413BE72397FBE78D3333E4C0F5C3ED55B09BE300BB53D7B6B05BE201B3BBEF85E8ABE50CB32BE4046C43D8C62B53D503FC93C7AF1B2BD90C061BEFAEFEABDB1A050BEF632DBBD9AE6253ED85D7B3D8428D8BD4E7C603E966F7A3E80A539BD3B9F813ED75187BE00869FBC3C5099BD5C0522BD2856DBBDD4EC5CBDE2786FBE391C833E1EA94A3E1824F13D8FAF6DBE2B11813E5EAE30BEF0911F3EAEEE8ABDE2E6353EB6FCD5BDE7128F3E0EAC283EE2569BBD50E7403DCA36713E907A243EA27C573E8A6B2A3ED6E37ABEB8DA4E3D3E0B57BE9A4C2F3E305FC8BC138B833E58926A3D3C40D23DD2A81A3E0EED633E136B87BEBE12713E1C436FBED40E0EBD56EE19BE95E366BE1078E7BD78F126BD4CFC193E5058CE3D1A7D673E3A2ACCBD00D344BB4E9BD9BD4CF1EFBD3CC48E3D4061253E06EF193EEE7DBDBD8644A6BDB83781BED50989BEE014A1BD6F02893E922934BE74AEC93D8886993D5AD991BD30E62EBE30D7B63C16365ABEDE30503E8006A73D29DE87BE00401CBCF0812BBE0E2357BED03EBB3DF7B2873EC8832E3D49328A3ED81D4EBD459989BE762C6D3E82702E3E787808BE84F3F9BD8004E73CC2E1723E5FEB803E20515E3E281D4F3EA2FC413E2CA163BE8AB6F7BDE84188BDF09D1BBDD542823E1423FD3DDA983C3E80123E3BF09E3C3D5C545A3E0FA708BEE2622F3E7E08D1BD5AA97D3E2852F83D40B8753C1A4C7C3EE4AD5CBE04B9923D3E6B783E1007533E44AEFCBDB4D8943D3633723E500F403E32C7003E1DB259BEA4F617BD9048A5BC469A643EA03B59BDEE5669BE6C5B2F3E672E4BBE40D9A6BBC6ED74BEE08FDC3CAB1929BE0AB683BD5D70883EB088AC3CDAF487BD7BA48BBEE461B93DA618F4BD5E5E32BE2F9919BE68CC823D7A974C3EEAB9293E1EAC4A3ED87A4FBE226A483ED4CE0F3E09CB8FBE082F703D50D9133E242DB1BDD25087BEE00AE0BC6C86173E0EAE013E348572BE8DC06BBECA2DF3BD1D5A8EBE8CAA29BEFC150ABD30583FBE1E6154BEA0F93A3C4413993D2030FE3CDEB41B3E7AD99DBDA0D4AA3C30BCC33DCCBB41BE61AD03BE7E85143E109197BC886786BE187C093E757202BEFC33313E84DAC6BD38C07CBDD79384BEB12461BE4B6F873E4A6C633E0891323EBDB8843E14AEC53D10C8903C1AB5323EE06B523C60E6803C8DEE8E3E80AE803D082E1D3DF0104BBE24050BBEF26C07BE8022FE3C90478F3D109E2BBD97F4833EC07285BE6D5A65BEDCE5CF3D08559EBDEE2F123E441FA93D10B4C3BCBFE7823E2E50663E24C3F5BD983E32BDCCED8BBEBA1C51BEF45B23BE5036A83CB058A1BC21FE863EA75D71BE763D44BE34CBFC3D829FACBD403C4EBDA00E223EE02EE4BC647AFB3D10DF773D61B225BE96615BBEE6DA1E3E3075F93D6666373E266368BEB0A01B3D905AE43C341E453EC016333CDE5731BEC0A608BDF6BB463E4CABD23D98AC46BE6A9D3B3EA2F4643ED26F36BEEC1CC63D702FE3BC10C3D9BD6031333D2446193E84269BBD8E217C3E13898ABE90F497BDC9F212BE5088D2BCC0D3773C90C09ABD433D13BE049244BEA63A023EC00B38BC98C1473DD027B3BD50D73B3D480A9C3D74F06DBD02EEFBBDA64C253EB4D9F13D6640773E28C4B7BDCE91783E367728BE939A80BE3C2D76BDB2D3223E7CB6D8BDE08B6B3C9A2EB7BD5886163E8404083E185C09BEBA7D463EE6777A3E624637BED87E01BDCC92D8BD3C69CCBD8C91DCBDA0C0183EA72D893E6022243E204B893CD19A85BE98F38ABD50A13F3EC08F6CBD2EF32EBEA1F404BE8069203E206E6ABC846D1FBEB81D543E4E8DCCBDB8E5633D7811A83D08A7AE3D80294B3BFE42523EC001ADBD5AEE3BBE004F233B3829353DE81067BD009DC63A1E8117BECC32B2BD43FD7DBEBE6E8BBE4295363E94D7B63D7CC22A3E804FABBD22842B3E28A7593E1C500DBDCC95FE3D3C059D3D805B233BCE75303E10FAE73CDAF2693E80D1083EBA4EDFBDF83791BD67F485BE76454FBEEC662FBECE6653BE4E1945BEE3DC8C3E903528BD7BF363BE00B2493E80BD843CE86BAB3D700299BD74374ABD7666543E00A73C3C60C7C53CB882D93D28519BBDC28D77BE777123BEF63674BEB15E80BEC2370D3ECC06A43D7C321D3EE603193E72716E3E70DAC83CEA485F3EE0E9513E32B4723ECC72D63DD2F914BE958602BEC4DB8EBED20DCDBD275E0ABEA19728BEE0396ABC7AD8233EF6A68EBE0044BBB95DD818BE2BA88FBEAAFDC3BD76BA753EA03D41BEE0DD653DD49A243EA38C50BE102BC0BCEA33723EC0216E3D4058823DA8BE29BD8CCC183EB2CA85BE0435603EA412093E6A3C87BE5E4C78BE005488BBA837753DCC7EF7BD803C3CBB3419343EC2D63DBE632787BE8C5184BE680B313E0D8E8C3E22F22B3E108CF0BC9BAC8CBE323E51BE2A4F073E04E9C4BD1428833D1AA50ABE98D4303E21C473BE3CE37DBE93DC8EBE085D173E203981BE3026353D5ABDDDBDB269D5BDA6D6153E66B6253E28E4CC3D10ED303D6C28373E709D093D40C5DABBD065433EF0BBA9BC64B5113EE2537E3EB30170BEB06B083E9CA1F93DF6F76E3E56DB7A3E90D387BE9A21303EA86881BD8A517C3EA9788EBE730C0CBE32A8D2BDDC12E33D7074A93D84B5F83D4C644A3E447AF63D44BB7DBD98FD133EE24A2C3E7030973C28A3383E40C4063DD03833BD83BA8E3E12E335BE908B533EE03EC13DF86C9B3D98922EBEA28140BED16A86BEDA2D49BEAA2E6A3E6007A0BC2CF5523E266DFBBD70F4123E5C8039BE6DAF13BE9079C0BCD0E85DBEA9577DBECA1C4C3ED063693D4651033E056E2FBEDE806DBE98226CBDA886373EB09FE93D4004003D9BC31CBEB14B3DBEF093DB3D540E103E90675F3DA2E44F3EDCFDC73D60054E3E98ECF0BD68F0043D7610013E4E8F543E26F357BED0A713BDA0E1573C90DD523EADA580BE6670A8BD2D558B3E805D61BE9E2719BE42FF35BE6887F03D806ADF3DB42A64BD00D56D3BFE0C643EE31857BEBC3667BEF065DBBC50C5B93DF8A8A5BDB7B380BE2A6E30BE10CF4E3DB44B0CBED7FA8CBE8E0A03BE4244663E31B85EBE701C74BDD1A46EBE8049F13D70C8C7BCC08A193D48A80F3D209A4A3C2C11213ED027DC3CDA41DABD466F50BEB1B41BBE84CFE8BDA8F802BE668B1E3E8E9D8BBD7C0E993D6E61223EB84DC73D6B657FBEE4B6973DF20D83BDCFDC1DBEBEA035BED4F701BEE642643E34435ABE7696663EC21A4C3EA228763EC8D30B3EF8DE99BDAB8C16BE629351BE1091C43D7E7B383EFCAC53BD02966DBE4C2E2D3E8F9E883E126F743E442F83BE507F4CBE59D5893EC044093CCFA01EBEF07D2C3E893909BE4288C5BDA612F4BDD0AA233DD43E813DF414223EF8C5893D38C7713D40E4FF3CD5874DBE88179EBD0807E53DC27A173E8465DEBD80EED43B4EE755BE2CB2843D1E678BBDF8497EBEE0874B3E002D0B3B6838073E954E0BBE809C07BEE2B1743E4A1D743E0C80203E0D98833E4A09563E36A089BD9C2897BD90B1E2BCC859283E846BB13DF0AF80BE2A6686BEC639F4BDDDDE87BEC2E4313E003A643B8A33073E8B47883E6098A2BDCCA46FBDBC63B63DA92D08BEAE1F103E42D2673E3AA3B3BDBE415A3E545AC13DFC6F88BD08AA05BEB066A2BC459715BE2C1F15BED6B5323ECD018CBEC51620BE0E07D9BD2EEFFEBD777D843EA027EB3CB87140BD28A8493D84632FBEDCDDE2BD929964BE8178823EBEDC713E98E458BDC6E14B3E80BB213B9299CEBD78F8263E40024A3EC036E7BCAF1E7FBE8829DF3DC00942BC66FC2E3EFA184FBEFA0930BE8C6ACF3D45D56FBE7827153D8400FBBD6E653BBE5411853D6CE580BE7C42C83DB4855EBED8F3483E7A1B723E70FDEB3C4B1C11BE107D373E663A173E"
    }
  }
#-}
