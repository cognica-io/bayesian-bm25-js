//
// Bayesian BM25
//
// Copyright (c) 2023-2026 Cognica, Inc.
//

// Calibration metrics for evaluating probability quality.
//
// Provides Expected Calibration Error (ECE), Brier score, and reliability
// diagram data for assessing how well predicted probabilities match actual
// relevance rates.

// Expected Calibration Error (ECE).
//
// Measures how well predicted probabilities match actual relevance
// rates.  Lower is better.  Perfect calibration = 0.
export function expectedCalibrationError(
  probabilities: number[],
  labels: number[],
  nBins: number = 10,
): number {
  const total = probabilities.length;
  const binWidth = 1.0 / nBins;
  let ece = 0.0;

  for (let bin = 0; bin < nBins; bin++) {
    const lo = bin * binWidth;
    const hi = (bin + 1) * binWidth;

    let sumProb = 0.0;
    let sumLabel = 0.0;
    let count = 0;

    for (let i = 0; i < total; i++) {
      const p = probabilities[i]!;
      const inBin =
        bin === 0 ? p >= lo && p <= hi : p > lo && p <= hi;
      if (inBin) {
        sumProb += p;
        sumLabel += labels[i]!;
        count++;
      }
    }

    if (count === 0) continue;
    const avgProb = sumProb / count;
    const avgLabel = sumLabel / count;
    ece += (count / total) * Math.abs(avgProb - avgLabel);
  }

  return ece;
}

// Brier score: mean squared error between probabilities and labels.
//
// Decomposes into calibration + discrimination.  Lower is better.
// A constant prediction of base rate achieves the reference score.
export function brierScore(
  probabilities: number[],
  labels: number[],
): number {
  let sum = 0.0;
  for (let i = 0; i < probabilities.length; i++) {
    const diff = probabilities[i]! - labels[i]!;
    sum += diff * diff;
  }
  return sum / probabilities.length;
}

// Reliability diagram data: [avgPredicted, avgActual, count] per bin.
//
// Perfect calibration means avgPredicted === avgActual for every bin.
export function reliabilityDiagram(
  probabilities: number[],
  labels: number[],
  nBins: number = 10,
): Array<[number, number, number]> {
  const binWidth = 1.0 / nBins;
  const bins: Array<[number, number, number]> = [];

  for (let bin = 0; bin < nBins; bin++) {
    const lo = bin * binWidth;
    const hi = (bin + 1) * binWidth;

    let sumProb = 0.0;
    let sumLabel = 0.0;
    let count = 0;

    for (let i = 0; i < probabilities.length; i++) {
      const p = probabilities[i]!;
      const inBin =
        bin === 0 ? p >= lo && p <= hi : p > lo && p <= hi;
      if (inBin) {
        sumProb += p;
        sumLabel += labels[i]!;
        count++;
      }
    }

    if (count === 0) continue;
    bins.push([sumProb / count, sumLabel / count, count]);
  }

  return bins;
}

// One-call calibration diagnostic report.
//
// Bundles ECE, Brier score, and reliability diagram data into a single
// object with a human-readable summary() method.
export class CalibrationReport {
  readonly ece: number;
  readonly brier: number;
  readonly reliability: Array<[number, number, number]>;
  readonly nSamples: number;
  readonly nBins: number;

  constructor(
    ece: number,
    brier: number,
    reliability: Array<[number, number, number]>,
    nSamples: number,
    nBins: number,
  ) {
    this.ece = ece;
    this.brier = brier;
    this.reliability = reliability;
    this.nSamples = nSamples;
    this.nBins = nBins;
  }

  // Formatted text summary of calibration metrics.
  summary(): string {
    const lines = [
      "Calibration Report",
      "==================",
      `  Samples : ${this.nSamples}`,
      `  Bins    : ${this.nBins}`,
      `  ECE     : ${this.ece.toFixed(6)}`,
      `  Brier   : ${this.brier.toFixed(6)}`,
      "",
      "  Reliability Diagram",
      "  -------------------",
      `  ${"Predicted".padStart(10)}  ${"Actual".padStart(10)}  ${"Count".padStart(6)}`,
    ];
    for (const [avgPred, avgActual, count] of this.reliability) {
      lines.push(
        `  ${avgPred.toFixed(4).padStart(10)}  ${avgActual.toFixed(4).padStart(10)}  ${String(count).padStart(6)}`,
      );
    }
    return lines.join("\n");
  }
}

// Compute a full calibration diagnostic report in one call.
export function calibrationReport(
  probabilities: number[],
  labels: number[],
  nBins: number = 10,
): CalibrationReport {
  const ece = expectedCalibrationError(probabilities, labels, nBins);
  const brier = brierScore(probabilities, labels);
  const reliability = reliabilityDiagram(probabilities, labels, nBins);

  return new CalibrationReport(
    ece,
    brier,
    reliability,
    probabilities.length,
    nBins,
  );
}
