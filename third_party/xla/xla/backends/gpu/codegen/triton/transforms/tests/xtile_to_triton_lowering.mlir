// RUN: xla-opt %s -split-input-file \
// RUN: -xtile-lower-to-triton \
// RUN: | FileCheck %s


// CHECK: func @lower_dot_scaled_add_to_triton(%[[LHS:.*]]: tensor<128x128xf8E5M2>, %[[LHS_SCALE:.*]]: tensor<128x4xi8>, %[[RHS:.*]]: tensor<128x256xf8E5M2>, %[[RHS_SCALE:.*]]: tensor<256x4xi8>, %[[ACC:.*]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
func.func @lower_dot_scaled_add_to_triton(
  %lhs: tensor<128x128xf8E5M2>, %lhs_scale: tensor<128x4xi8>,
  %rhs: tensor<128x256xf8E5M2>, %rhs_scale: tensor<256x4xi8>,
  %acc: tensor<128x256xf32>) -> tensor<128x256xf32> {
  // CHECK: %[[RES:.*]] = tt.dot_scaled %[[LHS]] scale %[[LHS_SCALE]], %[[RHS]] scale %[[RHS_SCALE]], %[[ACC]] lhs = e5m2 rhs = e5m2 {fastMath = true} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  // CHECK-NOT: arith.addf
  %0 = xtile.dot_scaled %lhs scale %lhs_scale, %rhs scale %rhs_scale
    {fastMath = true} : tensor<128x128xf8E5M2>,
    tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  %1 = arith.addf %acc, %0 : tensor<128x256xf32>
  // CHECK: return %[[RES]] : tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}

// -----

// CHECK-LABEL: func.func @scan_lowering(
// CHECK-SAME: %[[INPUT0:.*]]: tensor<16x16x16xf32>, %[[INPUT1:.*]]: tensor<16x16x16xf32>, %[[INIT0:.*]]: tensor<16x16xf32>, %[[INIT1:.*]]: tensor<16x16xf32>
func.func @scan_lowering(%input0: tensor<16x16x16xf32>, %input1: tensor<16x16x16xf32>, %init0: tensor<16x16xf32>, %init1: tensor<16x16xf32>) -> (tensor<16x16x16xf32>, tensor<16x16x16xf32>) {
  // CHECK: %[[SCAN:.*]]:2 = "tt.scan"(%[[INPUT0]], %[[INPUT1]]) <{axis = 2 : i32, reverse = false}> ({
  // CHECK:   ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32):
  // CHECK:     %[[TARG0:.*]] = tensor.from_elements %[[ARG0]] : tensor<1xf32>
  // CHECK:     %[[TARG1:.*]] = tensor.from_elements %[[ARG1]] : tensor<1xf32>
  // CHECK:     %[[TARG2:.*]] = tensor.from_elements %[[ARG2]] : tensor<1xf32>
  // CHECK:     %[[TARG3:.*]] = tensor.from_elements %[[ARG3]] : tensor<1xf32>
  // CHECK:     %[[ADD1:.*]] = stablehlo.add %[[TARG0]], %[[TARG2]] : tensor<1xf32>
  // CHECK:     %[[ADD2:.*]] = stablehlo.add %[[TARG1]], %[[TARG3]] : tensor<1xf32>
  // CHECK:     %[[EXT1:.*]] = tensor.extract %[[ADD1]]
  // CHECK:     %[[EXT2:.*]] = tensor.extract %[[ADD2]]
  // CHECK:     tt.scan.return %[[EXT1]], %[[EXT2]] : f32, f32
  // CHECK: }

  // CHECK-DAG: %[[BCAST_INIT0:.*]] = stablehlo.broadcast_in_dim %[[INIT0]], dims = [0, 1] : (tensor<16x16xf32>) -> tensor<16x16x16xf32>
  // CHECK-DAG: %[[BCAST_INIT1:.*]] = stablehlo.broadcast_in_dim %[[INIT1]], dims = [0, 1] : (tensor<16x16xf32>) -> tensor<16x16x16xf32>
  // CHECK-DAG: %[[RES1:.*]] = stablehlo.add %[[SCAN]]#0, %[[BCAST_INIT0]] : tensor<16x16x16xf32>
  // CHECK-DAG: %[[RES2:.*]] = stablehlo.add %[[SCAN]]#1, %[[BCAST_INIT1]] : tensor<16x16x16xf32>
  // CHECK: return %[[RES1]], %[[RES2]] : tensor<16x16x16xf32>, tensor<16x16x16xf32>

  %0, %1, %2, %3 = xtile.scan(%input0, %input1) inits(%init0, %init1) dimension = 2 {size = 16 : i64} : (tensor<16x16x16xf32>, tensor<16x16x16xf32>), (tensor<16x16xf32>, tensor<16x16xf32>) -> (tensor<16x16x16xf32>, tensor<16x16x16xf32>), (tensor<16x16xf32>, tensor<16x16xf32>) {
  ^bb0(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>):
    %add1 = stablehlo.add %arg0, %arg2 : tensor<1xf32>
    %add2 = stablehlo.add %arg1, %arg3 : tensor<1xf32>
    stablehlo.return %add1, %add2 : tensor<1xf32>, tensor<1xf32>
  }
  return %0, %1 : tensor<16x16x16xf32>, tensor<16x16x16xf32>
}

// CHECK: func @lower_dot_scaled_without_add_falls_back_to_xtile(%[[LHS:.*]]: tensor<128x128xf8E5M2>, %[[LHS_SCALE:.*]]: tensor<128x4xi8>, %[[RHS:.*]]: tensor<128x256xf8E5M2>, %[[RHS_SCALE:.*]]: tensor<256x4xi8>) -> tensor<128x256xf32> {
func.func @lower_dot_scaled_without_add_falls_back_to_xtile(
  %lhs: tensor<128x128xf8E5M2>, %lhs_scale: tensor<128x4xi8>,
  %rhs: tensor<128x256xf8E5M2>, %rhs_scale: tensor<256x4xi8>)
  -> tensor<128x256xf32> {
  // CHECK: %[[RES:.*]] = xtile.dot_scaled %[[LHS]] scale %[[LHS_SCALE]], %[[RHS]] scale %[[RHS_SCALE]] {fastMath = true} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  %0 = xtile.dot_scaled %lhs scale %lhs_scale, %rhs scale %rhs_scale
    {fastMath = true} : tensor<128x128xf8E5M2>,
    tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  // CHECK: return %[[RES]] : tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}