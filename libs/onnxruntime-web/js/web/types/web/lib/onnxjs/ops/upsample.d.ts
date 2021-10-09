import { Attribute } from '../attribute';
import { InferenceHandler } from '../backend';
import { Graph } from '../graph';
import { Operator } from '../operators';
import { Tensor } from '../tensor';
export declare abstract class Upsample implements Operator {
    protected opset: number;
    constructor(opset: number);
    abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[] | Promise<Tensor[]>;
    initialize(attributes: Attribute, _node: Graph.Node, _graph: Graph): void;
    checkInputs(inputs: Tensor[]): boolean;
    protected checkInputTypes(inputs: Tensor[]): boolean;
    protected prepareInputs(inputs: Tensor[]): [number[], number[], readonly number[]];
    protected isResize: boolean;
    protected mode: string;
    protected scales: number[];
    protected extrapolationValue: number;
    protected coordinateTransformMode: string;
    protected useExtrapolation: boolean;
    protected needRoiInput: boolean;
    protected nearestMode: string;
    protected cubicCoefficientA: number;
    protected excludeOutside: boolean;
    protected useNearest2xOptimization: boolean;
    protected roiInputIdx: number;
    protected scalesInputIdx: number;
    protected sizesInputIdx: number;
    protected roi: number[];
}
export declare function parseRoiData(roi: Tensor): number[];
export declare function parseScalesData(scale: Tensor, mode: string, isResize: boolean): number[];
export declare function parseScalesDataFromOutputSize(yDims: readonly number[], xDims: readonly number[], mode: string, isResize: boolean): number[];
export declare function computeOutputShape(scales: readonly number[], inputDims: readonly number[]): number[];
