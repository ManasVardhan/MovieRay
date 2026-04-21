import type { SegmentLabel } from "../types";
import { SEGMENT_DISPLAY_NAMES, SEGMENT_COLORS } from "../types";

interface ControlsProps {
  skipNonContent: boolean;
  onToggleSkipNonContent: () => void;
  skippedCategories: Set<SegmentLabel>;
  onToggleCategory: (label: SegmentLabel) => void;
  onPlayContentOnly: () => void;
}

const NON_CONTENT_LABELS: SegmentLabel[] = [
  "intro",
  "outro",
  "sponsorship",
  "self_promotion",
  "recap",
  "transition",
  "dead_air",
  "filler",
];

export function Controls({
  skipNonContent,
  onToggleSkipNonContent,
  skippedCategories,
  onToggleCategory,
  onPlayContentOnly,
}: ControlsProps) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 flex flex-wrap items-center gap-4">
      <button
        onClick={onPlayContentOnly}
        className="px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
      >
        Play Content Only
      </button>

      <button
        onClick={onToggleSkipNonContent}
        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
          skipNonContent
            ? "bg-red-600 hover:bg-red-500 text-white"
            : "bg-gray-700 hover:bg-gray-600 text-gray-300"
        }`}
      >
        {skipNonContent ? "Skip Non-Content: ON" : "Skip Non-Content: OFF"}
      </button>

      <div className="h-6 w-px bg-gray-700" />

      <div className="flex flex-wrap gap-2">
        {NON_CONTENT_LABELS.map((label) => (
          <label
            key={label}
            className="flex items-center gap-1.5 text-xs text-gray-300 cursor-pointer"
          >
            <input
              type="checkbox"
              checked={skippedCategories.has(label)}
              onChange={() => onToggleCategory(label)}
              className="rounded border-gray-600"
            />
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: SEGMENT_COLORS[label] }}
            />
            {SEGMENT_DISPLAY_NAMES[label]}
          </label>
        ))}
      </div>
    </div>
  );
}
