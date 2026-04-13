import { readFileSync } from "node:fs";
import { resolve } from "node:path";

export type FigureEntry = {
  dataset: string;
  model: string;
  temperature: number;
  dflashSpeedup: number;
  ddtreeSpeedup: number;
};

const DATASET_ORDER = [
  "MATH-500",
  "GSM8K",
  "AIME 2024",
  "AIME 2025",
  "HumanEval",
  "MBPP",
  "LiveCodeBench",
  "SWE-bench Lite",
  "MT-Bench",
  "Alpaca",
];

const MODEL_ORDER = [
  "Qwen3-4B",
  "Qwen3-8B",
  "Qwen3-Coder-30B-A3B-Instruct",
];

function stripLatex(value: string): string {
  return value
    .replace(/\\textbf\{([^}]*)\}/g, "$1")
    .replace(/\$\\times\$/g, "")
    .replace(/\$/g, "")
    .replace(/\\\\/g, "")
    .trim();
}

function parseSpeedup(value: string): number {
  return Number.parseFloat(stripLatex(value).replace("×", "").trim());
}

function loadResultsTable(): string {
  const candidates = [resolve(process.cwd(), "src", "data", "results.tex")];

  for (const path of candidates) {
    try {
      return readFileSync(path, "utf8");
    } catch {
      continue;
    }
  }

  return "";
}

export function getFigureOverviewEntries(): FigureEntry[] {
  const content = loadResultsTable();
  if (!content) {
    return [];
  }

  const lines = content.split("\n");
  const entries: FigureEntry[] = [];
  let currentTemperature: number | null = null;

  for (const line of lines) {
    if (line.includes("Temperature = 0.0")) {
      currentTemperature = 0.0;
      continue;
    }

    if (line.includes("Temperature = 1.0")) {
      currentTemperature = 1.0;
      continue;
    }

    if (currentTemperature === null || !line.includes("&") || line.trim().startsWith("\\")) {
      continue;
    }

    const columns = line
      .split("&")
      .map((value) => value.trim())
      .map(stripLatex);

    if (columns.length !== 13) {
      continue;
    }

    const dataset = columns[0]!;
    const values = columns.slice(1);

    for (let modelIndex = 0; modelIndex < MODEL_ORDER.length; modelIndex += 1) {
      const offset = modelIndex * 4;
      entries.push({
        dataset,
        model: MODEL_ORDER[modelIndex]!,
        temperature: currentTemperature,
        dflashSpeedup: parseSpeedup(values[offset]!),
        ddtreeSpeedup: parseSpeedup(values[offset + 2]!),
      });
    }
  }

  return entries.sort((left, right) => {
    const temperatureDelta = left.temperature - right.temperature;
    if (temperatureDelta !== 0) {
      return temperatureDelta;
    }

    const datasetDelta = DATASET_ORDER.indexOf(left.dataset) - DATASET_ORDER.indexOf(right.dataset);
    if (datasetDelta !== 0) {
      return datasetDelta;
    }

    return MODEL_ORDER.indexOf(left.model) - MODEL_ORDER.indexOf(right.model);
  });
}
