"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Upload, X, Check, Loader2 } from "lucide-react";

export function PhotoUploader() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0];
      if (selectedFile) {
        setFile(selectedFile);
        setError(null);

        // Create preview
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreview(reader.result as string);
        };
        reader.readAsDataURL(selectedFile);
      }
    },
    []
  );

  const handleUploadAndAnalyze = async () => {
    if (!file) return;

    try {
      setError(null);
      setUploading(true);
      setProgress(20);

      const formData = new FormData();
      formData.append("file", file);

      const uploadResponse = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        throw new Error(errorData.error || "Upload failed");
      }

      const { url } = await uploadResponse.json();
      setProgress(40);
      setUploading(false);
      setAnalyzing(true);

      const analyzeResponse = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageUrl: url }),
      });

      if (!analyzeResponse.ok) {
        const errorData = await analyzeResponse.json();
        throw new Error(errorData.error || "Analysis failed");
      }

      const { analysisId } = await analyzeResponse.json();
      setProgress(100);

      setTimeout(() => {
        router.push(`/analysis/${analysisId}`);
      }, 500);
    } catch (err: any) {
      console.error("Error:", err);
      setError(err.message || "An error occurred");
      setUploading(false);
      setAnalyzing(false);
      setProgress(0);
    }
  };

  const clearSelection = () => {
    setFile(null);
    setPreview(null);
    setError(null);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      setFile(droppedFile);
      setError(null);

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(droppedFile);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const isProcessing = uploading || analyzing;

  return (
    <div className="w-full max-w-3xl mx-auto">
      <Card className="overflow-hidden border-2 border-dashed hover:border-primary/50 transition-colors">
        {!preview ? (
          <div
            className="relative p-12 md:p-16 text-center cursor-pointer group"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <div className="space-y-6">
              <div className="flex justify-center">
                <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                  <Upload className="w-10 h-10 text-primary" />
                </div>
              </div>

              <div className="space-y-2">
                <h3 className="text-2xl font-bold">Upload your photo</h3>
                <p className="text-muted-foreground max-w-sm mx-auto">
                  Drag and drop your image here, or click to browse
                </p>
              </div>

              <div className="flex flex-wrap justify-center gap-2 text-xs text-muted-foreground">
                <span className="px-3 py-1 rounded-full bg-muted">JPG</span>
                <span className="px-3 py-1 rounded-full bg-muted">PNG</span>
                <span className="px-3 py-1 rounded-full bg-muted">Max 5MB</span>
              </div>
            </div>

            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
        ) : (
          <div className="p-6 space-y-6">
            <div className="relative group">
              <img
                src={preview}
                alt="Preview"
                className="w-full h-96 object-cover rounded-lg"
              />
              {!isProcessing && (
                <button
                  onClick={clearSelection}
                  className="absolute top-4 right-4 h-10 w-10 rounded-full bg-background/90 backdrop-blur flex items-center justify-center hover:bg-background shadow-lg opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X className="h-5 w-5" />
                </button>
              )}
            </div>

            {isProcessing && (
              <Card className="p-6 bg-muted/50">
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-2">
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      <span className="font-medium">
                        {uploading && "Uploading your photo..."}
                        {analyzing && "Analyzing phenotypes..."}
                      </span>
                    </div>
                    <span className="text-muted-foreground">{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                  <p className="text-xs text-muted-foreground text-center">
                    This may take 15-30 seconds
                  </p>
                </div>
              </Card>
            )}

            {error && (
              <Card className="p-4 bg-destructive/10 border-destructive">
                <p className="text-sm text-destructive text-center">{error}</p>
              </Card>
            )}

            {!isProcessing && (
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  onClick={clearSelection}
                  className="flex-1"
                >
                  Choose Different Photo
                </Button>
                <Button onClick={handleUploadAndAnalyze} className="flex-1" size="lg">
                  <Check className="w-4 h-4 mr-2" />
                  Analyze Photo
                </Button>
              </div>
            )}
          </div>
        )}
      </Card>

      <div className="mt-8 grid md:grid-cols-3 gap-4 text-sm">
        <div className="text-center">
          <div className="font-semibold mb-1">Fast Analysis</div>
          <div className="text-muted-foreground">Results in 30 seconds</div>
        </div>
        <div className="text-center">
          <div className="font-semibold mb-1">AI-Powered</div>
          <div className="text-muted-foreground">512-dim embeddings</div>
        </div>
        <div className="text-center">
          <div className="font-semibold mb-1">Private & Secure</div>
          <div className="text-muted-foreground">Encrypted storage</div>
        </div>
      </div>
    </div>
  );
}
