import { NextRequest, NextResponse } from "next/server";
import { put } from "@vercel/blob";
import { stackServerApp } from "@/app/stack";
import { generateImageEmbedding } from "@/lib/embeddings";
import { neon } from "@neondatabase/serverless";

const connection = neon(process.env.DATABASE_URL!);

// Simple admin check - in production, you'd want proper role-based auth
const ADMIN_EMAILS = [
  "admin@example.com",
  // Add your admin email here
];

export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const user = await stackServerApp.getUser();
    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    // Check if user is admin (simple check - enhance for production)
    // For now, we'll allow any authenticated user to add phenotypes
    // TODO: Add proper admin role check

    const formData = await request.formData();
    const file = formData.get("file") as File;
    const name = formData.get("name") as string;
    const description = formData.get("description") as string;
    const geographicOrigin = formData.get("geographicOrigin") as string;
    const keyTraits = formData.get("keyTraits") as string; // JSON string

    if (!file || !name) {
      return NextResponse.json(
        { error: "File and name are required" },
        { status: 400 }
      );
    }

    // Validate file type
    if (!file.type.startsWith("image/")) {
      return NextResponse.json(
        { error: "File must be an image" },
        { status: 400 }
      );
    }

    // Upload image to Vercel Blob
    const filename = `phenotypes/${Date.now()}-${file.name}`;
    const blob = await put(filename, file, {
      access: "public",
    });

    // Generate embedding for the phenotype image
    console.log("Generating embedding for phenotype image...");
    const embedding = await generateImageEmbedding(blob.url);
    const embeddingString = `[${embedding.join(",")}]`;

    // Parse key traits if provided
    let traitsJson = null;
    if (keyTraits) {
      try {
        traitsJson = JSON.parse(keyTraits);
      } catch (e) {
        // If not valid JSON, treat as comma-separated string
        traitsJson = keyTraits.split(",").map((t) => t.trim());
      }
    }

    // Insert into database
    const result = await connection`
      INSERT INTO phenotypes (
        name,
        description,
        geographic_origin,
        key_traits,
        image_url,
        embedding,
        created_at
      )
      VALUES (
        ${name},
        ${description || null},
        ${geographicOrigin || null},
        ${traitsJson ? JSON.stringify(traitsJson) : null}::jsonb,
        ${blob.url},
        ${embeddingString}::vector,
        NOW()
      )
      RETURNING id, name
    `;

    return NextResponse.json({
      success: true,
      phenotype: result[0],
    });
  } catch (error: any) {
    console.error("Phenotype creation error:", error);
    return NextResponse.json(
      {
        error: "Failed to create phenotype",
        details: error.message || "Unknown error",
      },
      { status: 500 }
    );
  }
}

// Get all phenotypes
export async function GET(request: NextRequest) {
  try {
    const user = await stackServerApp.getUser();
    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const phenotypes = await connection`
      SELECT
        id,
        name,
        description,
        geographic_origin as "geographicOrigin",
        key_traits as "keyTraits",
        image_url as "imageUrl",
        created_at as "createdAt"
      FROM phenotypes
      ORDER BY created_at DESC
    `;

    return NextResponse.json({ phenotypes });
  } catch (error) {
    console.error("Phenotype fetch error:", error);
    return NextResponse.json(
      { error: "Failed to fetch phenotypes" },
      { status: 500 }
    );
  }
}
