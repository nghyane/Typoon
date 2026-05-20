export { K, InpaintJob, InpaintsDonePayload } from "@typoon/shared";

export interface PreparedPage {
  index:  number;
  width:  number;
  height: number;
}

export interface PreparedChapterMeta {
  chapter_id:  number;
  strategy:    "one_to_one" | "stitch";
  is_color:    boolean;
  color_ratio: number;
  pages:       PreparedPage[];
  groups:      number[][];
  raw_count:   number;
}

/** Message enqueued to typoon-scan-queue per page. */
export interface ScanJob {
  workflow_id:  string;
  chapter_id:   number;
  page_index:   number;
  total_pages:  number;
  prepared_key: string;
  is_color:     boolean;
}

/** Payload sent via instance.sendEvent("scans-done", ...) */
export interface ScansDonePayload {
  scan_keys: string[];
  mask_keys: string[];
}
