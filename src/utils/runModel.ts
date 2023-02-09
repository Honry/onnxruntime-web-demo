import {InferenceSession, Tensor, env} from 'onnxruntime-web';

function init() {
  env.wasm.proxy = true;
  env.wasm.numThreads = 1;
  env.wasm.simd = true;
}

export async function createModelCpu(model: ArrayBuffer): Promise<InferenceSession> {
  init();
  return await InferenceSession.create(model, {executionProviders: ['wasm']});
}
export async function createModelGpu(model: ArrayBuffer): Promise<InferenceSession> {
  init();
  return await InferenceSession.create(model, {executionProviders: ['webgl']});
}
export async function createModelWebnn(model: ArrayBuffer, devicePreference: number = 0): Promise<InferenceSession> {
  // Build version
  env.wasm.wasmPaths = {
    "ort-wasm.wasm": location.origin + "/onnxruntime-web-demo/js/ort-wasm.wasm",
    "ort-wasm-simd.wasm": location.origin + "/onnxruntime-web-demo/js/ort-wasm-simd.wasm",
    "ort-wasm-threaded.wasm": location.origin + "/onnxruntime-web-demo/js/ort-wasm-threaded.wasm",
    "ort-wasm-simd-threaded.wasm":
        location.origin + "/onnxruntime-web-demo/js/ort-wasm-simd-threaded.wasm",
  };

  // Dev version
  // env.wasm.wasmPaths = {
  //     "ort-wasm.wasm": location.origin + "/js/ort-wasm.wasm",
  //     "ort-wasm-simd.wasm": location.origin + "/js/ort-wasm-simd.wasm",
  //     "ort-wasm-threaded.wasm": location.origin + "/js/ort-wasm-threaded.wasm",
  //     "ort-wasm-simd-threaded.wasm":
  //         location.origin + "/js/ort-wasm-simd-threaded.wasm",
  // };
  return await InferenceSession.create(model, {executionProviders: ['wasm', {name: 'webnn', deviceType: devicePreference}], logSeverityLevel: 0});
}

export async function warmupModel(model: InferenceSession, dims: number[]) {
  // OK. we generate a random input and call Session.run() as a warmup query
  const size = dims.reduce((a, b) => a * b);
  const warmupTensor = new Tensor('float32', new Float32Array(size), dims);

  for (let i = 0; i < size; i++) {
    warmupTensor.data[i] = Math.random() * 2.0 - 1.0;  // random value [-1.0, 1.0)
  }
  try {
    const feeds: Record<string, Tensor> = {};
    feeds[model.inputNames[0]] = warmupTensor;
    const start = performance.now();
    await model.run(feeds);
    console.log(`warm up time: ${(performance.now() - start).toFixed(2)} ms`);
  } catch (e) {
    console.error(e);
  }
}

export async function runModel(model: InferenceSession, preprocessedData: Tensor): Promise<[Tensor, number]> {
  try {
    let inferenceTime;
    let output: any;
    let inferenceTimeArray = [];
    const numRuns = getNumRuns();
    console.log(`Start inference for ${numRuns} times...`);
    for (let i = 0; i < numRuns; i ++) {
      const newTensor = new Tensor(preprocessedData.type, preprocessedData.data.slice(0), preprocessedData.dims);
      const feeds: Record<string, Tensor> = {};
      feeds[model.inputNames[0]] = newTensor;
      const start = performance.now();
      // console.time("model inference");
      const outputData = await model.run(feeds);
      // console.timeEnd("model inference");
      inferenceTime = performance.now() - start;
      inferenceTimeArray.push(inferenceTime);
      output = outputData[model.outputNames[0]];
      console.log(`num ${i+1} inference time: ${inferenceTime.toFixed(2)}`);
    }
    inferenceTime = getMedianValue(inferenceTimeArray);
    console.log(`Done... median inference time is ${inferenceTime.toFixed(2)} ms`);
    return [output, inferenceTime];
  } catch (e) {
    console.error(e);
    throw new Error();
  }
}

// Get median value from an array of Number
function getMedianValue(array: Array<any>) {
  array = array.sort((a, b) => a - b);
  return array.length % 2 !== 0 ? array[Math.floor(array.length / 2)] :
      (array[array.length / 2 - 1] + array[array.length / 2]) / 2;
}

function getNumRuns() {
  const params = new URLSearchParams(location.hash.split('?')[1]);
  let numRunsParam = params.get('numRuns');
  if (numRunsParam && Number.isInteger(parseInt(numRunsParam)) && parseInt(numRunsParam) > 0) {
    return parseInt(numRunsParam);
  } else {
    return 1;
  }
}