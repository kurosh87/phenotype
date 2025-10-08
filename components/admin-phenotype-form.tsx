"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useRouter } from "next/navigation";

export function AdminPhenotypeForm() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    geographicOrigin: "",
    keyTraits: "",
  });
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setError("Please select an image");
      return;
    }

    if (!formData.name) {
      setError("Phenotype name is required");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setSuccess(false);
      setProgress(20);

      const submitFormData = new FormData();
      submitFormData.append("file", file);
      submitFormData.append("name", formData.name);
      submitFormData.append("description", formData.description);
      submitFormData.append("geographicOrigin", formData.geographicOrigin);
      submitFormData.append("keyTraits", formData.keyTraits);

      setProgress(40);

      const response = await fetch("/api/admin/phenotypes", {
        method: "POST",
        body: submitFormData,
      });

      setProgress(80);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to create phenotype");
      }

      setProgress(100);
      setSuccess(true);

      // Reset form
      setFormData({
        name: "",
        description: "",
        geographicOrigin: "",
        keyTraits: "",
      });
      setFile(null);
      setPreview(null);

      // Refresh the page to show new phenotype
      router.refresh();

      setTimeout(() => {
        setSuccess(false);
        setProgress(0);
      }, 3000);
    } catch (err: any) {
      console.error("Error:", err);
      setError(err.message || "An error occurred");
      setProgress(0);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Add New Phenotype</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Image Upload */}
          <div className="space-y-2">
            <Label htmlFor="image">Phenotype Image *</Label>
            {preview ? (
              <div className="relative">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-48 object-cover rounded-lg"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setFile(null);
                    setPreview(null);
                  }}
                  className="mt-2"
                >
                  Change Image
                </Button>
              </div>
            ) : (
              <Input
                id="image"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                disabled={loading}
              />
            )}
          </div>

          {/* Name */}
          <div className="space-y-2">
            <Label htmlFor="name">Phenotype Name *</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) =>
                setFormData({ ...formData, name: e.target.value })
              }
              placeholder="e.g., Mediterranean"
              disabled={loading}
              required
            />
          </div>

          {/* Description */}
          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <textarea
              id="description"
              value={formData.description}
              onChange={(e) =>
                setFormData({ ...formData, description: e.target.value })
              }
              placeholder="Detailed description of this phenotype..."
              disabled={loading}
              className="w-full min-h-[100px] px-3 py-2 border rounded-md"
            />
          </div>

          {/* Geographic Origin */}
          <div className="space-y-2">
            <Label htmlFor="geographicOrigin">Geographic Origin</Label>
            <Input
              id="geographicOrigin"
              value={formData.geographicOrigin}
              onChange={(e) =>
                setFormData({ ...formData, geographicOrigin: e.target.value })
              }
              placeholder="e.g., Southern Europe, Mediterranean Region"
              disabled={loading}
            />
          </div>

          {/* Key Traits */}
          <div className="space-y-2">
            <Label htmlFor="keyTraits">Key Traits (comma-separated)</Label>
            <Input
              id="keyTraits"
              value={formData.keyTraits}
              onChange={(e) =>
                setFormData({ ...formData, keyTraits: e.target.value })
              }
              placeholder="e.g., olive skin, dark hair, almond eyes"
              disabled={loading}
            />
          </div>

          {/* Progress Bar */}
          {loading && (
            <div className="space-y-2">
              <Progress value={progress} />
              <p className="text-sm text-muted-foreground text-center">
                {progress < 40 && "Uploading image..."}
                {progress >= 40 && progress < 80 && "Generating embedding..."}
                {progress >= 80 && "Saving to database..."}
              </p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-destructive/10 text-destructive px-4 py-3 rounded-md text-sm">
              {error}
            </div>
          )}

          {/* Success Message */}
          {success && (
            <div className="bg-green-500/10 text-green-600 px-4 py-3 rounded-md text-sm">
              Phenotype added successfully!
            </div>
          )}

          {/* Submit Button */}
          <Button type="submit" disabled={loading} className="w-full">
            {loading ? "Adding Phenotype..." : "Add Phenotype"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
