"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.extractTransferableBuffers = exports.endProfiling = exports.run = exports.releaseSession = exports.createSession = exports.initOrt = void 0;
const run_options_1 = require("./run-options");
const session_options_1 = require("./session-options");
const string_utils_1 = require("./string-utils");
const wasm_factory_1 = require("./wasm-factory");
/**
 * initialize ORT environment.
 * @param numThreads SetGlobalIntraOpNumThreads(numThreads)
 * @param loggingLevel CreateEnv(static_cast<OrtLoggingLevel>(logging_level))
 */
const initOrt = (numThreads, loggingLevel) => {
    const errorCode = wasm_factory_1.getInstance()._OrtInit(numThreads, loggingLevel);
    if (errorCode !== 0) {
        throw new Error(`Can't initialize onnxruntime. error code = ${errorCode}`);
    }
};
exports.initOrt = initOrt;
const activeSessions = [];
/**
 * create an instance of InferenceSession.
 * @returns the metadata of InferenceSession. 0-value handle for failure.
 */
const createSession = (model, options) => {
    const wasm = wasm_factory_1.getInstance();
    const modelDataOffset = wasm._malloc(model.byteLength);
    let sessionHandle = 0;
    let sessionOptionsHandle = 0;
    let allocs = [];
    try {
        [sessionOptionsHandle, allocs] = session_options_1.setSessionOptions(options);
        wasm.HEAPU8.set(model, modelDataOffset);
        sessionHandle = wasm._OrtCreateSession(modelDataOffset, model.byteLength, sessionOptionsHandle);
        if (sessionHandle === 0) {
            throw new Error('Can\'t create a session');
        }
    }
    finally {
        wasm._free(modelDataOffset);
        wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
        allocs.forEach(wasm._free);
    }
    const inputCount = wasm._OrtGetInputCount(sessionHandle);
    const outputCount = wasm._OrtGetOutputCount(sessionHandle);
    const inputNames = [];
    const inputNamesUTF8Encoded = [];
    const outputNames = [];
    const outputNamesUTF8Encoded = [];
    for (let i = 0; i < inputCount; i++) {
        const name = wasm._OrtGetInputName(sessionHandle, i);
        if (name === 0) {
            throw new Error('Can\'t get an input name');
        }
        inputNamesUTF8Encoded.push(name);
        inputNames.push(wasm.UTF8ToString(name));
    }
    for (let i = 0; i < outputCount; i++) {
        const name = wasm._OrtGetOutputName(sessionHandle, i);
        if (name === 0) {
            throw new Error('Can\'t get an output name');
        }
        outputNamesUTF8Encoded.push(name);
        outputNames.push(wasm.UTF8ToString(name));
    }
    activeSessions.push([sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded]);
    return [activeSessions.length - 1, inputNames, outputNames];
};
exports.createSession = createSession;
const releaseSession = (sessionId) => {
    const wasm = wasm_factory_1.getInstance();
    const session = activeSessions[sessionId];
    if (!session) {
        throw new Error('invalid session id');
    }
    const sessionHandle = session[0];
    const inputNamesUTF8Encoded = session[1];
    const outputNamesUTF8Encoded = session[2];
    inputNamesUTF8Encoded.forEach(wasm._OrtFree);
    outputNamesUTF8Encoded.forEach(wasm._OrtFree);
    wasm._OrtReleaseSession(sessionHandle);
    activeSessions[sessionId] = undefined;
};
exports.releaseSession = releaseSession;
const tensorDataTypeStringToEnum = (type) => {
    switch (type) {
        case 'int8':
            return 3 /* int8 */;
        case 'uint8':
            return 2 /* uint8 */;
        case 'bool':
            return 9 /* bool */;
        case 'int16':
            return 5 /* int16 */;
        case 'uint16':
            return 4 /* uint16 */;
        case 'int32':
            return 6 /* int32 */;
        case 'uint32':
            return 12 /* uint32 */;
        case 'float32':
            return 1 /* float */;
        case 'float64':
            return 11 /* double */;
        case 'string':
            return 8 /* string */;
        case 'int64':
            return 7 /* int64 */;
        case 'uint64':
            return 13 /* uint64 */;
        default:
            throw new Error(`unsupported data type: ${type}`);
    }
};
const tensorDataTypeEnumToString = (typeProto) => {
    switch (typeProto) {
        case 3 /* int8 */:
            return 'int8';
        case 2 /* uint8 */:
            return 'uint8';
        case 9 /* bool */:
            return 'bool';
        case 5 /* int16 */:
            return 'int16';
        case 4 /* uint16 */:
            return 'uint16';
        case 6 /* int32 */:
            return 'int32';
        case 12 /* uint32 */:
            return 'uint32';
        case 1 /* float */:
            return 'float32';
        case 11 /* double */:
            return 'float64';
        case 8 /* string */:
            return 'string';
        case 7 /* int64 */:
            return 'int32';
        case 13 /* uint64 */:
            return 'uint32';
        default:
            throw new Error(`unsupported data type: ${typeProto}`);
    }
};
const numericTensorTypeToTypedArray = (type) => {
    switch (type) {
        case 'float32':
            return Float32Array;
        case 'uint8':
            return Uint8Array;
        case 'int8':
            return Int8Array;
        case 'uint16':
            return Uint16Array;
        case 'int16':
            return Int16Array;
        case 'int32':
            return Int32Array;
        case 'bool':
            return Uint8Array;
        case 'float64':
            return Float64Array;
        case 'uint32':
            return Uint32Array;
        case 'int64':
            return BigInt64Array;
        case 'uint64':
            return BigUint64Array;
        default:
            throw new Error(`unsupported type: ${type}`);
    }
};
/**
 * perform inference run
 */
const run = (sessionId, inputIndices, inputs, outputIndices, options) => {
    const wasm = wasm_factory_1.getInstance();
    const session = activeSessions[sessionId];
    if (!session) {
        throw new Error('invalid session id');
    }
    const sessionHandle = session[0];
    const inputNamesUTF8Encoded = session[1];
    const outputNamesUTF8Encoded = session[2];
    const inputCount = inputIndices.length;
    const outputCount = outputIndices.length;
    let runOptionsHandle = 0;
    let runOptionsAllocs = [];
    const inputValues = [];
    const inputAllocs = [];
    try {
        [runOptionsHandle, runOptionsAllocs] = run_options_1.setRunOptions(options);
        // create input tensors
        for (let i = 0; i < inputCount; i++) {
            const dataType = inputs[i][0];
            const dims = inputs[i][1];
            const data = inputs[i][2];
            let dataOffset;
            let dataByteLength;
            if (Array.isArray(data)) {
                // string tensor
                dataByteLength = 4 * data.length;
                dataOffset = wasm._malloc(dataByteLength);
                inputAllocs.push(dataOffset);
                let dataIndex = dataOffset / 4;
                for (let i = 0; i < data.length; i++) {
                    if (typeof data[i] !== 'string') {
                        throw new TypeError(`tensor data at index ${i} is not a string`);
                    }
                    wasm.HEAPU32[dataIndex++] = string_utils_1.allocWasmString(data[i], inputAllocs);
                }
            }
            else {
                dataByteLength = data.byteLength;
                dataOffset = wasm._malloc(dataByteLength);
                inputAllocs.push(dataOffset);
                wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, dataByteLength), dataOffset);
            }
            const stack = wasm.stackSave();
            const dimsOffset = wasm.stackAlloc(4 * dims.length);
            try {
                let dimIndex = dimsOffset / 4;
                dims.forEach(d => wasm.HEAP32[dimIndex++] = d);
                const tensor = wasm._OrtCreateTensor(tensorDataTypeStringToEnum(dataType), dataOffset, dataByteLength, dimsOffset, dims.length);
                if (tensor === 0) {
                    throw new Error('Can\'t create a tensor');
                }
                inputValues.push(tensor);
            }
            finally {
                wasm.stackRestore(stack);
            }
        }
        const beforeRunStack = wasm.stackSave();
        const inputValuesOffset = wasm.stackAlloc(inputCount * 4);
        const inputNamesOffset = wasm.stackAlloc(inputCount * 4);
        const outputValuesOffset = wasm.stackAlloc(outputCount * 4);
        const outputNamesOffset = wasm.stackAlloc(outputCount * 4);
        try {
            let inputValuesIndex = inputValuesOffset / 4;
            let inputNamesIndex = inputNamesOffset / 4;
            let outputValuesIndex = outputValuesOffset / 4;
            let outputNamesIndex = outputNamesOffset / 4;
            for (let i = 0; i < inputCount; i++) {
                wasm.HEAPU32[inputValuesIndex++] = inputValues[i];
                wasm.HEAPU32[inputNamesIndex++] = inputNamesUTF8Encoded[inputIndices[i]];
            }
            for (let i = 0; i < outputCount; i++) {
                wasm.HEAPU32[outputValuesIndex++] = 0;
                wasm.HEAPU32[outputNamesIndex++] = outputNamesUTF8Encoded[outputIndices[i]];
            }
            // support RunOptions
            let errorCode = wasm._OrtRun(sessionHandle, inputNamesOffset, inputValuesOffset, inputCount, outputNamesOffset, outputCount, outputValuesOffset, runOptionsHandle);
            const output = [];
            if (errorCode === 0) {
                for (let i = 0; i < outputCount; i++) {
                    const tensor = wasm.HEAPU32[outputValuesOffset / 4 + i];
                    const beforeGetTensorDataStack = wasm.stackSave();
                    // stack allocate 4 pointer value
                    const tensorDataOffset = wasm.stackAlloc(4 * 4);
                    let type, dataOffset = 0;
                    try {
                        errorCode = wasm._OrtGetTensorData(tensor, tensorDataOffset, tensorDataOffset + 4, tensorDataOffset + 8, tensorDataOffset + 12);
                        if (errorCode !== 0) {
                            throw new Error(`Can't get a tensor data. error code = ${errorCode}`);
                        }
                        let tensorDataIndex = tensorDataOffset / 4;
                        const dataType = wasm.HEAPU32[tensorDataIndex++];
                        dataOffset = wasm.HEAPU32[tensorDataIndex++];
                        const dimsOffset = wasm.HEAPU32[tensorDataIndex++];
                        const dimsLength = wasm.HEAPU32[tensorDataIndex++];
                        const dims = [];
                        for (let i = 0; i < dimsLength; i++) {
                            dims.push(wasm.HEAPU32[dimsOffset / 4 + i]);
                        }
                        wasm._OrtFree(dimsOffset);
                        const size = dims.length === 0 ? 1 : dims.reduce((a, b) => a * b);
                        type = tensorDataTypeEnumToString(dataType);
                        if (type === 'string') {
                            const stringData = [];
                            let dataIndex = dataOffset / 4;
                            for (let i = 0; i < size; i++) {
                                const offset = wasm.HEAPU32[dataIndex++];
                                const maxBytesToRead = i === size - 1 ? undefined : wasm.HEAPU32[dataIndex] - offset;
                                stringData.push(wasm.UTF8ToString(offset, maxBytesToRead));
                            }
                            output.push([type, dims, stringData]);
                        }
                        else {
                            const typedArrayConstructor = numericTensorTypeToTypedArray(type);
                            const data = new typedArrayConstructor(size);
                            new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                                .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + data.byteLength));
                            output.push([type, dims, data]);
                        }
                    }
                    finally {
                        wasm.stackRestore(beforeGetTensorDataStack);
                        if (type === 'string' && dataOffset) {
                            wasm._free(dataOffset);
                        }
                        wasm._OrtReleaseTensor(tensor);
                    }
                }
            }
            if (errorCode === 0) {
                return output;
            }
            else {
                throw new Error(`failed to call OrtRun(). error code = ${errorCode}.`);
            }
        }
        finally {
            wasm.stackRestore(beforeRunStack);
        }
    }
    finally {
        inputValues.forEach(wasm._OrtReleaseTensor);
        inputAllocs.forEach(wasm._free);
        wasm._OrtReleaseRunOptions(runOptionsHandle);
        runOptionsAllocs.forEach(wasm._free);
    }
};
exports.run = run;
/**
 * end profiling
 */
const endProfiling = (sessionId) => {
    const wasm = wasm_factory_1.getInstance();
    const session = activeSessions[sessionId];
    if (!session) {
        throw new Error('invalid session id');
    }
    const sessionHandle = session[0];
    // profile file name is not used yet, but it must be freed.
    const profileFileName = wasm._OrtEndProfiling(sessionHandle);
    if (profileFileName === 0) {
        throw new Error('Can\'t get an profile file name');
    }
    wasm._OrtFree(profileFileName);
};
exports.endProfiling = endProfiling;
const extractTransferableBuffers = (tensors) => {
    const buffers = [];
    for (const tensor of tensors) {
        const data = tensor[2];
        if (!Array.isArray(data) && data.buffer) {
            buffers.push(data.buffer);
        }
    }
    return buffers;
};
exports.extractTransferableBuffers = extractTransferableBuffers;
//# sourceMappingURL=wasm-core-impl.js.map