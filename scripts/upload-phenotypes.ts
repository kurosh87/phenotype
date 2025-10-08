#!/usr/bin/env tsx
/**
 * Upload Phenotypes Script
 * Uploads scraped phenotype data to Vercel Blob and Neon Database
 *
 * Usage: tsx scripts/upload-phenotypes.ts <path-to-phenotypes.json>
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { put } from "@vercel/blob";
import { neon } from "@neondatabase/serverless";
import { generateImageEmbedding } from "../lib/embeddings";
import fs from "fs/promises";
import path from "path";

if (!process.env.DATABASE_URL) {
  console.error("❌ DATABASE_URL environment variable is not set");
  process.exit(1);
}

if (!process.env.BLOB_READ_WRITE_TOKEN) {
  console.error("❌ BLOB_READ_WRITE_TOKEN environment variable is not set");
  process.exit(1);
}

if (!process.env.REPLICATE_API_TOKEN) {
  console.warn("⚠️  REPLICATE_API_TOKEN not set - embeddings will be skipped");
  console.warn("   (You can add embeddings later by running the embedding generation script)");
}

const connection = neon(process.env.DATABASE_URL);

interface ScrapedPhenotype {
  url: string;
  name: string;
  phenotype_groups: string[];
  description: string | null;
  physical_traits: string | null;
  literature: Array<{ text: string; url?: string }>;
  similar_types: Array<{ name: string; url: string }>;
  male_image_url: string | null;
  male_image_local: string | null;
  female_image_url: string | null;
  female_image_local: string | null;
  map_image_url: string | null;
  map_image_local: string | null;
}

interface PhenotypeRecord {
  id: string;
  name: string;
  originalUrl: string;
}

async function uploadImageToBlob(
  localPath: string,
  phenotypeName: string,
  imageType: string
): Promise<string | null> {
  try {
    const fileBuffer = await fs.readFile(localPath);
    const fileName = `phenotypes/${phenotypeName.replace(/\s+/g, "_")}/${imageType}_${Date.now()}${path.extname(localPath)}`;

    const blob = await put(fileName, fileBuffer, {
      access: "public",
    });

    console.log(`  ✓ Uploaded ${imageType} to ${blob.url}`);
    return blob.url;
  } catch (error) {
    console.error(`  ✗ Failed to upload ${imageType}:`, error);
    return null;
  }
}

async function generateEmbedding(imageUrl: string): Promise<number[] | null> {
  // Skip embeddings if Replicate token is not configured
  if (!process.env.REPLICATE_API_TOKEN) {
    console.log(`  ⊘ Skipping embedding generation (REPLICATE_API_TOKEN not set)`);
    return null;
  }

  try {
    const embedding = await generateImageEmbedding(imageUrl);
    console.log(`  ✓ Generated embedding (${embedding.length} dimensions)`);
    return embedding;
  } catch (error) {
    console.error(`  ✗ Failed to generate embedding:`, error);
    return null;
  }
}

async function insertPhenotype(
  data: ScrapedPhenotype,
  maleImageUrl: string | null,
  femaleImageUrl: string | null,
  mapImageUrl: string | null,
  embedding: number[] | null
): Promise<string | null> {
  try {
    const embeddingString = embedding ? `[${embedding.join(",")}]` : null;

    // Use male image as primary, fallback to female
    const primaryImageUrl = maleImageUrl || femaleImageUrl;

    if (!primaryImageUrl) {
      console.error(`  ✗ No images available for ${data.name}`);
      return null;
    }

    const result = await connection`
      INSERT INTO phenotypes (
        name,
        phenotype_groups,
        description,
        physical_traits,
        geographic_origin,
        literature,
        male_image_url,
        female_image_url,
        map_image_url,
        image_url,
        embedding,
        metadata,
        created_at
      )
      VALUES (
        ${data.name},
        ${data.phenotype_groups.length > 0 ? JSON.stringify(data.phenotype_groups) : null}::jsonb,
        ${data.description},
        ${data.physical_traits},
        ${null},
        ${data.literature.length > 0 ? JSON.stringify(data.literature) : null}::jsonb,
        ${maleImageUrl},
        ${femaleImageUrl},
        ${mapImageUrl},
        ${primaryImageUrl},
        ${embeddingString ? `${embeddingString}::vector` : null},
        ${JSON.stringify({ original_url: data.url })}::jsonb,
        NOW()
      )
      RETURNING id, name
    `;

    console.log(`  ✓ Inserted into database with ID: ${result[0].id}`);
    return result[0].id;
  } catch (error) {
    console.error(`  ✗ Failed to insert into database:`, error);
    return null;
  }
}

async function insertSimilarPhenotypes(
  phenotypeId: string,
  phenotypeName: string,
  similarTypes: Array<{ name: string; url: string }>,
  phenotypeMap: Map<string, PhenotypeRecord>
) {
  for (const similar of similarTypes) {
    // Find the similar phenotype in our map
    const similarRecord = phenotypeMap.get(similar.url);

    if (similarRecord) {
      try {
        await connection`
          INSERT INTO similar_phenotypes (phenotype_id, similar_phenotype_id, created_at)
          VALUES (${phenotypeId}, ${similarRecord.id}, NOW())
          ON CONFLICT DO NOTHING
        `;
        console.log(`    ✓ Linked ${phenotypeName} → ${similarRecord.name}`);
      } catch (error) {
        console.error(`    ✗ Failed to link ${phenotypeName} → ${similar.name}:`, error);
      }
    }
  }
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.error("Usage: tsx scripts/upload-phenotypes.ts <path-to-phenotypes.json>");
    process.exit(1);
  }

  const phenotypesFile = path.resolve(args[0]);
  const baseDir = path.dirname(phenotypesFile);

  console.log(`\n📂 Reading phenotypes from: ${phenotypesFile}`);
  console.log(`📁 Base directory: ${baseDir}\n`);

  // Read the scraped data
  const rawData = await fs.readFile(phenotypesFile, "utf-8");
  const scrapedData: Record<string, ScrapedPhenotype> = JSON.parse(rawData);

  // Resolve relative image paths
  for (const phenotype of Object.values(scrapedData)) {
    if (phenotype.male_image_local) {
      phenotype.male_image_local = path.resolve(baseDir, phenotype.male_image_local);
    }
    if (phenotype.female_image_local) {
      phenotype.female_image_local = path.resolve(baseDir, phenotype.female_image_local);
    }
    if (phenotype.map_image_local) {
      phenotype.map_image_local = path.resolve(baseDir, phenotype.map_image_local);
    }
  }

  const phenotypes = Object.values(scrapedData);
  console.log(`Found ${phenotypes.length} phenotypes to upload\n`);

  // Phase 1: Upload images and insert phenotypes
  console.log("=== PHASE 1: Uploading Images & Creating Records ===\n");

  const phenotypeMap = new Map<string, PhenotypeRecord>();
  let successCount = 0;
  let failCount = 0;

  for (let i = 0; i < phenotypes.length; i++) {
    const phenotype = phenotypes[i];
    console.log(`[${i + 1}/${phenotypes.length}] Processing: ${phenotype.name}`);

    // Upload images to Vercel Blob
    let maleImageUrl: string | null = null;
    let femaleImageUrl: string | null = null;
    let mapImageUrl: string | null = null;

    if (phenotype.male_image_local) {
      maleImageUrl = await uploadImageToBlob(
        phenotype.male_image_local,
        phenotype.name,
        "male"
      );
    }

    if (phenotype.female_image_local) {
      femaleImageUrl = await uploadImageToBlob(
        phenotype.female_image_local,
        phenotype.name,
        "female"
      );
    }

    if (phenotype.map_image_local) {
      mapImageUrl = await uploadImageToBlob(
        phenotype.map_image_local,
        phenotype.name,
        "map"
      );
    }

    // Generate embedding from primary image (prefer male)
    let embedding: number[] | null = null;
    const primaryImageUrl = maleImageUrl || femaleImageUrl;

    if (primaryImageUrl) {
      embedding = await generateEmbedding(primaryImageUrl);
    }

    // Insert into database
    const phenotypeId = await insertPhenotype(
      phenotype,
      maleImageUrl,
      femaleImageUrl,
      mapImageUrl,
      embedding
    );

    if (phenotypeId) {
      phenotypeMap.set(phenotype.url, {
        id: phenotypeId,
        name: phenotype.name,
        originalUrl: phenotype.url,
      });
      successCount++;
    } else {
      failCount++;
    }

    console.log("");
  }

  // Phase 2: Create similarity relationships
  console.log("\n=== PHASE 2: Creating Similarity Relationships ===\n");

  let relationshipCount = 0;

  for (const phenotype of phenotypes) {
    const record = phenotypeMap.get(phenotype.url);

    if (record && phenotype.similar_types.length > 0) {
      console.log(`Linking ${record.name} to ${phenotype.similar_types.length} similar phenotypes...`);
      await insertSimilarPhenotypes(
        record.id,
        record.name,
        phenotype.similar_types,
        phenotypeMap
      );
      relationshipCount += phenotype.similar_types.length;
    }
  }

  // Summary
  console.log("\n=== UPLOAD COMPLETE ===\n");
  console.log(`✓ Successfully uploaded: ${successCount} phenotypes`);
  console.log(`✗ Failed: ${failCount} phenotypes`);
  console.log(`✓ Created: ${relationshipCount} similarity relationships`);
  console.log("");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
