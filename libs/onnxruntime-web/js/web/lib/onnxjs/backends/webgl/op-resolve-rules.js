"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.WEBGL_OP_RESOLVE_RULES = void 0;
const batch_normalization_1 = require("./ops/batch-normalization");
const binaryOps = __importStar(require("./ops/binary-op"));
const concat_1 = require("./ops/concat");
const conv_1 = require("./ops/conv");
const depth_to_space_1 = require("./ops/depth-to-space");
const flatten_1 = require("./ops/flatten");
const gather_1 = require("./ops/gather");
const gemm_1 = require("./ops/gemm");
const image_scaler_1 = require("./ops/image-scaler");
const instance_normalization_1 = require("./ops/instance-normalization");
const matmul_1 = require("./ops/matmul");
const pad_1 = require("./ops/pad");
const pool_1 = require("./ops/pool");
const pool_2 = require("./ops/pool");
const pool_3 = require("./ops/pool");
const pool_4 = require("./ops/pool");
const reduce_1 = require("./ops/reduce");
const reduce_2 = require("./ops/reduce");
const reshape_1 = require("./ops/reshape");
const resize_packed_1 = require("./ops/resize-packed");
const shape_1 = require("./ops/shape");
const slice_1 = require("./ops/slice");
const softmax_1 = require("./ops/softmax");
const split_1 = require("./ops/split");
const squeeze_1 = require("./ops/squeeze");
const sum_1 = require("./ops/sum");
const tile_1 = require("./ops/tile");
const transpose_1 = require("./ops/transpose");
const unaryOps = __importStar(require("./ops/unary-op"));
const unsqueeze_1 = require("./ops/unsqueeze");
const upsample_1 = require("./ops/upsample");
exports.WEBGL_OP_RESOLVE_RULES = [
    ['Abs', '', '6+', unaryOps.abs],
    ['Acos', '', '7+', unaryOps.acos],
    ['Add', '', '7+', binaryOps.add],
    ['And', '', '7+', binaryOps.and],
    ['Asin', '', '7+', unaryOps.asin],
    ['Atan', '', '7+', unaryOps.atan],
    // TODO: support new attributes for AveragePool-10
    ['AveragePool', '', '7-10', pool_1.averagePool, pool_1.parseAveragePoolAttributes],
    ['BatchNormalization', '', '7+', batch_normalization_1.batchNormalization, batch_normalization_1.parseBatchNormalizationAttributes],
    ['Ceil', '', '6+', unaryOps.ceil],
    ['Clip', '', '6-10', unaryOps.clip, unaryOps.parseClipAttributes],
    ['Concat', '', '4+', concat_1.concat, concat_1.parseConcatAttributes],
    ['Conv', '', '1+', conv_1.conv, conv_1.parseConvAttributes],
    ['Cos', '', '7+', unaryOps.cos],
    ['Div', '', '7+', binaryOps.div],
    ['Dropout', '', '7+', unaryOps.identity],
    ['DepthToSpace', '', '1+', depth_to_space_1.depthToSpace, depth_to_space_1.parseDepthToSpaceAttributes],
    ['Equal', '', '7+', binaryOps.equal],
    ['Elu', '', '6+', unaryOps.elu, unaryOps.parseEluAttributes],
    ['Exp', '', '6+', unaryOps.exp],
    ['Flatten', '', '1+', flatten_1.flatten, flatten_1.parseFlattenAttributes],
    ['Floor', '', '6+', unaryOps.floor],
    ['Gather', '', '1+', gather_1.gather, gather_1.parseGatherAttributes],
    ['Gemm', '', '7-10', gemm_1.gemm, gemm_1.parseGemmAttributesV7],
    ['Gemm', '', '11+', gemm_1.gemm, gemm_1.parseGemmAttributesV11],
    ['GlobalAveragePool', '', '1+', pool_2.globalAveragePool, pool_2.parseGlobalAveragePoolAttributes],
    ['GlobalMaxPool', '', '1+', pool_4.globalMaxPool],
    ['Greater', '', '7+', binaryOps.greater],
    ['Identity', '', '1+', unaryOps.identity],
    ['ImageScaler', '', '1+', image_scaler_1.imageScaler, image_scaler_1.parseImageScalerAttributes],
    ['InstanceNormalization', '', '6+', instance_normalization_1.instanceNormalization, instance_normalization_1.parseInstanceNormalizationAttributes],
    ['LeakyRelu', '', '6+', unaryOps.leakyRelu, unaryOps.parseLeakyReluAttributes],
    ['Less', '', '7+', binaryOps.less],
    ['Log', '', '6+', unaryOps.log],
    ['MatMul', '', '1+', matmul_1.matMul, matmul_1.parseMatMulAttributes],
    // TODO: support new attributes for MaxPool-8 and MaxPool-10
    ['MaxPool', '', '1-9', pool_3.maxPool, pool_3.parseMaxPoolAttributes],
    ['Mul', '', '7+', binaryOps.mul],
    ['Neg', '', '6+', unaryOps.neg],
    ['Not', '', '1+', unaryOps.not],
    ['Or', '', '7+', binaryOps.or],
    ['Pad', '', '2-10', pad_1.pad, pad_1.parsePadAttributes],
    ['Pow', '', '7+', binaryOps.pow],
    ['PRelu', '', '7+', binaryOps.pRelu],
    ['ReduceLogSum', '', '1+', reduce_1.reduceLogSum, reduce_2.parseReduceAttributes],
    ['ReduceMax', '', '1+', reduce_1.reduceMax, reduce_2.parseReduceAttributes],
    ['ReduceMean', '', '1+', reduce_1.reduceMean, reduce_2.parseReduceAttributes],
    ['ReduceMin', '', '1+', reduce_1.reduceMin, reduce_2.parseReduceAttributes],
    ['ReduceProd', '', '1+', reduce_1.reduceProd, reduce_2.parseReduceAttributes],
    ['ReduceSum', '', '1+', reduce_1.reduceSum, reduce_2.parseReduceAttributes],
    ['ReduceSumSquare', '', '1+', reduce_1.reduceLogSumSquare, reduce_2.parseReduceAttributes],
    ['Relu', '', '6+', unaryOps.relu],
    ['Reshape', '', '5+', reshape_1.reshape],
    ['Resize', '', '10', resize_packed_1.resize, resize_packed_1.parseResizeAttributesV10],
    ['Resize', '', '11+', resize_packed_1.resize, resize_packed_1.parseResizeAttributesV11],
    ['Shape', '', '1+', shape_1.shape],
    ['Sigmoid', '', '6+', unaryOps.sigmoid],
    ['Sin', '', '7+', unaryOps.sin],
    ['Slice', '', '10+', slice_1.sliceV10],
    ['Slice', '', '1-9', slice_1.slice, slice_1.parseSliceAttributes],
    ['Softmax', '', '1+', softmax_1.softmax, softmax_1.parseSoftmaxAttributes],
    // 'Split' operator has an optional attribute 'split'
    // this attribute determines how the specified axis of input data is split.
    // When the attribute is missing, we need the count of number of outputs
    // so that we can determine the 'split' attribute from the runtime input to the Operator
    ['Split', '', '2+', split_1.split, split_1.parseSplitAttributes],
    ['Sqrt', '', '6+', unaryOps.sqrt],
    ['Squeeze', '', '1+', squeeze_1.squeeze, squeeze_1.parseSqueezeAttributes],
    ['Sub', '', '7+', binaryOps.sub],
    ['Sum', '', '6+', sum_1.sum],
    ['Tan', '', '7+', unaryOps.tan],
    ['Tanh', '', '6+', unaryOps.tanh],
    ['Tile', '', '6+', tile_1.tile],
    ['Transpose', '', '1+', transpose_1.transpose, transpose_1.parseTransposeAttributes],
    ['Upsample', '', '7-8', upsample_1.upsample, upsample_1.parseUpsampleAttributesV7],
    ['Upsample', '', '9', upsample_1.upsample, upsample_1.parseUpsampleAttributesV9],
    ['Unsqueeze', '', '1+', unsqueeze_1.unsqueeze, unsqueeze_1.parseUnsqueezeAttributes],
    ['Xor', '', '7+', binaryOps.xor],
];
//# sourceMappingURL=op-resolve-rules.js.map