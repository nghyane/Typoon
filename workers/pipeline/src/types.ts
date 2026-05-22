/** Re-exports for pipeline workflow.
 *  Keep types thin; everything else lives in `@typoon/shared`. */

export { K, type PreparedJobMeta } from "@typoon/shared";

export interface PreparedPage {
  index:  number;
  width:  number;
  height: number;
}
