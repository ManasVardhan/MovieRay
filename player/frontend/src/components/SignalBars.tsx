import { useRef, useEffect } from "react";
import type { SignalData } from "../types";

interface SignalBarsProps {
  signals: SignalData;
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}

interface BarConfig {
  label: string;
  color: string;
  data: number[];
}

export function SignalBars({
  signals,
  duration,
  currentTime,
  onSeek,
}: SignalBarsProps) {
  const audioCanvasRef = useRef<HTMLCanvasElement>(null);
  const motionCanvasRef = useRef<HTMLCanvasElement>(null);
  const staticCanvasRef = useRef<HTMLCanvasElement>(null);

  const bars: Array<{ ref: React.RefObject<HTMLCanvasElement | null>; config: BarConfig }> = [
    {
      ref: audioCanvasRef,
      config: {
        label: "Audio Energy",
        color: "#22d3ee",
        data: signals.audio_energy,
      },
    },
    {
      ref: motionCanvasRef,
      config: {
        label: "Motion",
        color: "#a78bfa",
        data: signals.motion,
      },
    },
    {
      ref: staticCanvasRef,
      config: {
        label: "Static Score",
        color: "#fb923c",
        data: signals.static_score,
      },
    },
  ];

  useEffect(() => {
    for (const { ref, config } of bars) {
      const canvas = ref.current;
      if (!canvas) continue;

      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;
      const data = config.data;
      const maxVal = Math.max(...data, 0.001);

      // Clear
      ctx.clearRect(0, 0, w, h);

      // Draw bars
      const barCount = data.length;
      const barWidth = w / barCount;

      for (let i = 0; i < barCount; i++) {
        const val = data[i] / maxVal;
        const barHeight = val * h;
        const x = i * barWidth;

        // Bar from bottom
        ctx.fillStyle = config.color + "88"; // semi-transparent
        ctx.fillRect(x, h - barHeight, Math.max(barWidth - 0.5, 0.5), barHeight);
      }

      // Draw playhead
      const playheadX = (currentTime / duration) * w;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(playheadX - 0.5, 0, 1, h);
    }
  }, [signals, currentTime, duration]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / rect.width;
    onSeek(ratio * duration);
  };

  return (
    <div className="bg-gray-900 rounded-lg p-4 space-y-3">
      <h3 className="text-white font-semibold text-sm uppercase tracking-wide">
        Signal Analysis
      </h3>

      {bars.map(({ ref, config }) => (
        <div key={config.label}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-400">{config.label}</span>
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: config.color }}
            />
          </div>
          <canvas
            ref={ref}
            className="w-full h-10 rounded cursor-pointer bg-gray-800"
            onClick={handleClick}
          />
        </div>
      ))}
    </div>
  );
}
