import { useEffect, useRef } from "react";
import type { Segment } from "../types";
import { SEGMENT_COLORS, SEGMENT_DISPLAY_NAMES } from "../types";

interface SegmentPanelProps {
  segments: Segment[];
  currentTime: number;
  onPlay: (segment: Segment) => void;
  onSkip: (segment: Segment) => void;
}

export function SegmentPanel({
  segments,
  currentTime,
  onPlay,
  onSkip,
}: SegmentPanelProps) {
  const activeRef = useRef<HTMLDivElement>(null);

  const activeIndex = segments.findIndex(
    (s) => currentTime >= s.start && currentTime < s.end
  );

  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [activeIndex]);

  return (
    <div className="bg-gray-900 rounded-lg p-4 h-full overflow-y-auto">
      <h3 className="text-white font-semibold mb-3 text-sm uppercase tracking-wide">
        Segment Overview
      </h3>

      <div className="space-y-2">
        {segments.map((seg, i) => {
          const isActive = i === activeIndex;
          const isLowConfidence = seg.confidence < 0.7;

          return (
            <div
              key={i}
              ref={isActive ? activeRef : null}
              className={`rounded-lg p-3 transition-all ${
                isActive
                  ? "bg-gray-700 ring-1 ring-white/30"
                  : "bg-gray-800 hover:bg-gray-750"
              } ${isLowConfidence ? "border border-dashed border-yellow-500/50" : ""}`}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: SEGMENT_COLORS[seg.label] }}
                  />
                  <span className="text-white text-sm font-medium">
                    {SEGMENT_DISPLAY_NAMES[seg.label]}
                  </span>
                  {isLowConfidence && (
                    <span className="text-yellow-500 text-xs" title="Low confidence">
                      ?
                    </span>
                  )}
                </div>
                <span className="text-gray-400 text-xs">
                  {formatTime(seg.start)} - {formatTime(seg.end)}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span
                  className={`text-xs px-2 py-0.5 rounded ${
                    seg.type === "content"
                      ? "bg-green-900/50 text-green-400"
                      : "bg-red-900/50 text-red-400"
                  }`}
                >
                  {seg.type === "content" ? "Content" : "Non-Content"}
                </span>

                <div className="flex gap-1">
                  <button
                    onClick={() => onPlay(seg)}
                    className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white transition-colors"
                  >
                    Play
                  </button>
                  <button
                    onClick={() => onSkip(seg)}
                    className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-white transition-colors"
                  >
                    Skip
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
