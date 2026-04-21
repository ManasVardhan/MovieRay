import { useState } from "react";
import {
  type Segment,
  SEGMENT_COLORS,
  SEGMENT_DISPLAY_NAMES,
} from "../types";

interface SegmentTimelineProps {
  segments: Segment[];
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}

export function SegmentTimeline({
  segments,
  duration,
  currentTime,
  onSeek,
}: SegmentTimelineProps) {
  const [hoveredSegment, setHoveredSegment] = useState<Segment | null>(null);
  const [tooltipX, setTooltipX] = useState(0);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / rect.width;
    onSeek(ratio * duration);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / rect.width;
    const time = ratio * duration;
    setTooltipX(x);

    const seg = segments.find((s) => time >= s.start && time < s.end);
    setHoveredSegment(seg ?? null);
  };

  const playheadPosition = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="relative">
      {hoveredSegment && (
        <div
          className="absolute bottom-full mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap pointer-events-none z-10"
          style={{ left: tooltipX, transform: "translateX(-50%)" }}
        >
          <span className="font-medium">
            {SEGMENT_DISPLAY_NAMES[hoveredSegment.label]}
          </span>
          <span className="text-gray-400 ml-2">
            {formatTime(hoveredSegment.start)} - {formatTime(hoveredSegment.end)}
          </span>
        </div>
      )}

      <div
        className="relative h-8 rounded-md overflow-hidden cursor-pointer flex"
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredSegment(null)}
      >
        {segments.map((seg, i) => {
          const widthPct = ((seg.end - seg.start) / duration) * 100;
          const isActive = currentTime >= seg.start && currentTime < seg.end;

          return (
            <div
              key={i}
              className={`h-full transition-opacity ${
                isActive ? "ring-2 ring-white ring-inset" : ""
              }`}
              style={{
                width: `${widthPct}%`,
                backgroundColor: SEGMENT_COLORS[seg.label],
                opacity: hoveredSegment === seg ? 1 : 0.8,
              }}
            />
          );
        })}

        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
          style={{ left: `${playheadPosition}%` }}
        />
      </div>

      <div className="flex mt-1 text-xs text-gray-500">
        {segments.map((seg, i) => {
          const widthPct = ((seg.end - seg.start) / duration) * 100;
          if (widthPct < 5) return <div key={i} style={{ width: `${widthPct}%` }} />;
          return (
            <div
              key={i}
              className="truncate text-center"
              style={{ width: `${widthPct}%` }}
            >
              {SEGMENT_DISPLAY_NAMES[seg.label]}
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
