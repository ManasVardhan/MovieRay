import { useState, useRef, useCallback, useEffect } from "react";
import type { VideoPlayerHandle } from "./components/VideoPlayer";
import { VideoPlayer } from "./components/VideoPlayer";
import { SegmentTimeline } from "./components/SegmentTimeline";
import { SegmentPanel } from "./components/SegmentPanel";
import { Controls } from "./components/Controls";
import type {
  Segment,
  SegmentLabel,
  VideoMetadata,
  VideoListItem,
} from "./types";

function App() {
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [skipNonContent, setSkipNonContent] = useState(false);
  const [skippedCategories, setSkippedCategories] = useState<Set<SegmentLabel>>(
    new Set(["sponsorship", "dead_air"])
  );

  const playerRef = useRef<VideoPlayerHandle>(null);

  useEffect(() => {
    fetch("/api/videos")
      .then((r) => r.json())
      .then((data) => setVideos(data.videos))
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedVideoId) return;
    fetch(`/api/videos/${selectedVideoId}/metadata`)
      .then((r) => r.json())
      .then((data: VideoMetadata) => setMetadata(data))
      .catch(console.error);
  }, [selectedVideoId]);

  const handleTimeUpdate = useCallback(
    (time: number) => {
      setCurrentTime(time);

      if (!skipNonContent || !metadata) return;

      const currentSegment = metadata.segments.find(
        (s) => time >= s.start && time < s.end
      );

      if (
        currentSegment &&
        currentSegment.type === "non-content" &&
        skippedCategories.has(currentSegment.label)
      ) {
        playerRef.current?.seek(currentSegment.end);
      }
    },
    [skipNonContent, metadata, skippedCategories]
  );

  const handleSeek = (time: number) => {
    playerRef.current?.seek(time);
  };

  const handlePlaySegment = (seg: Segment) => {
    playerRef.current?.seek(seg.start);
    playerRef.current?.play();
  };

  const handleSkipSegment = (seg: Segment) => {
    playerRef.current?.seek(seg.end);
  };

  const handlePlayContentOnly = () => {
    setSkipNonContent(true);
    setSkippedCategories(
      new Set<SegmentLabel>([
        "intro", "outro", "sponsorship", "self_promotion",
        "recap", "transition", "dead_air", "filler",
      ])
    );
    playerRef.current?.seek(0);
    playerRef.current?.play();
  };

  const handleToggleCategory = (label: SegmentLabel) => {
    setSkippedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  };

  if (!selectedVideoId) {
    return (
      <div className="min-h-screen bg-gray-950 text-white p-8">
        <h1 className="text-3xl font-bold mb-2">MovieRay</h1>
        <p className="text-gray-400 mb-8">
          Multimodal Video Segmentation Player
        </p>

        {videos.length === 0 ? (
          <div className="text-gray-500">
            <p>No analyzed videos found.</p>
            <p className="text-sm mt-2">
              Run{" "}
              <code className="bg-gray-800 px-2 py-1 rounded">
                python pipeline/analyze.py video.mp4
              </code>{" "}
              to analyze a video first.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {videos.map((v) => (
              <button
                key={v.id}
                onClick={() => setSelectedVideoId(v.id)}
                className="text-left bg-gray-900 rounded-lg p-4 hover:bg-gray-800 transition-colors"
              >
                <h3 className="font-medium mb-1">{v.name}</h3>
                <p className="text-gray-400 text-sm">
                  {Math.floor(v.duration / 60)}m {Math.floor(v.duration % 60)}s
                  &middot; {v.segment_count} segments
                </p>
                {!v.has_video_file && (
                  <p className="text-yellow-500 text-xs mt-1">
                    Video file missing
                  </p>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  const videoUrl = `/api/videos/${selectedVideoId}/stream`;
  const segments = metadata?.segments ?? [];

  return (
    <div className="min-h-screen bg-gray-950 text-white p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => {
              setSelectedVideoId(null);
              setMetadata(null);
            }}
            className="text-gray-400 hover:text-white transition-colors"
          >
            &larr; Back
          </button>
          <h1 className="text-xl font-bold">
            {metadata?.video ?? selectedVideoId}
          </h1>
        </div>
        <span className="text-gray-500 text-sm">
          Processed by MovieRay
        </span>
      </div>

      <div className="flex gap-4">
        <div className="flex-1 space-y-3">
          <VideoPlayer
            ref={playerRef}
            videoUrl={videoUrl}
            onTimeUpdate={handleTimeUpdate}
            onDurationChange={setDuration}
          />

          <SegmentTimeline
            segments={segments}
            duration={duration}
            currentTime={currentTime}
            onSeek={handleSeek}
          />

          <Controls
            skipNonContent={skipNonContent}
            onToggleSkipNonContent={() => setSkipNonContent((p) => !p)}
            skippedCategories={skippedCategories}
            onToggleCategory={handleToggleCategory}
            onPlayContentOnly={handlePlayContentOnly}
          />
        </div>

        <div className="w-80 flex-shrink-0">
          <SegmentPanel
            segments={segments}
            currentTime={currentTime}
            onPlay={handlePlaySegment}
            onSkip={handleSkipSegment}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
