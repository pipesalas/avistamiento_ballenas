export const temperatureRamp = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026"];
export const chlorophyllRamp = ["#edf8e5", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b"];

export function rampFor(variable: string): string[] {
  return variable === "temperature" ? temperatureRamp : chlorophyllRamp;
}
