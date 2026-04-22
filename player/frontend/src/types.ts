export interface Segment {
  start: number;
  end: number;
  label: SegmentLabel;
  type: "content" | "non-content";
  confidence: number;
  reason: string;
}

export type SegmentLabel =
  | "core_content"
  | "intro"
  | "outro"
  | "sponsorship"
  | "self_promotion"
  | "recap"
  | "transition"
  | "dead_air"
  | "filler";

export interface SignalData {
  timestamps: number[];
  audio_energy: number[];
  motion: number[];
  static_score: number[];
}

export interface VideoMetadata {
  video: string;
  duration: number;
  analyzed_at: string;
  segments: Segment[];
  signals?: SignalData;
}

export interface VideoListItem {
  id: string;
  name: string;
  duration: number;
  segment_count: number;
  has_video_file: boolean;
}

export const SEGMENT_COLORS: Record<SegmentLabel, string> = {
  core_content: "#22c55e",
  intro: "#3b82f6",
  outro: "#6366f1",
  sponsorship: "#ef4444",
  self_promotion: "#f97316",
  recap: "#eab308",
  transition: "#8b5cf6",
  dead_air: "#6b7280",
  filler: "#a855f7",
};

export const SEGMENT_DISPLAY_NAMES: Record<SegmentLabel, string> = {
  core_content: "Content",
  intro: "Intro",
  outro: "Outro",
  sponsorship: "Sponsorship",
  self_promotion: "Self Promo",
  recap: "Recap",
  transition: "Transition",
  dead_air: "Dead Air",
  filler: "Filler",
};
