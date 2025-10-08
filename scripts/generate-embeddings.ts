#!/usr/bin/env tsx
/**
 * Generate Embeddings for Existing Phenotypes
 * Updates phenotypes that don't have embeddings yet
 *
 * Usage: tsx scripts/generate-embeddings.ts
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { neon } from "@neondatabase/serverless";
import { generateImageEmbedding } from "../lib/embeddings";

if (!process.env.DATABASE_URL) {
  console.error("❌ DATABASE_URL environment variable is not set");
  process.exit(1);
}

if (!process.env.REPLICATE_API_TOKEN) {
  console.error("❌ REPLICATE_API_TOKEN environment variable is not set");
  process.exit(1);
}

const sql = neon(process.env.DATABASE_URL);

interface Phenotype {
  id: string;
  name: string;
  image_url: string;
  male_image_url: string | null;
  female_image_url: string | null;
}

async function main() {
  console.log("\n🔄 Starting embedding generation for existing phenotypes...\n");

  // Get all phenotypes without embeddings
  const result = await sql`
    SELECT id, name, image_url, male_image_url, female_image_url
    FROM phenotypes
    WHERE embedding IS NULL
    ORDER BY name ASC
  `;
  const phenotypes = result as Phenotype[];

  if (phenotypes.length === 0) {
    console.log("✅ All phenotypes already have embeddings!");
    return;
  }

  console.log(`Found ${phenotypes.length} phenotypes without embeddings\n`);

  let successCount = 0;
  let errorCount = 0;

  for (let i = 0; i < phenotypes.length; i++) {
    const phenotype = phenotypes[i];
    console.log(`[${i + 1}/${phenotypes.length}] Processing: ${phenotype.name}`);

    try {
      // Prioritize male image, fall back to female, then main image
      const imageUrl = phenotype.male_image_url || phenotype.female_image_url || phenotype.image_url;

      if (!imageUrl) {
        console.log(`  ⚠️  No image URL found, skipping`);
        errorCount++;
        continue;
      }

      // Generate embedding
      const embedding = await generateImageEmbedding(imageUrl);

      if (!embedding || embedding.length === 0) {
        console.log(`  ✗ Invalid embedding generated`);
        errorCount++;
        continue;
      }

      // Convert to PostgreSQL vector format
      const embeddingString = `[${embedding.join(",")}]`;

      // Update database
      await sql`
        UPDATE phenotypes
        SET embedding = ${embeddingString}::vector
        WHERE id = ${phenotype.id}
      `;

      console.log(`  ✓ Generated ${embedding.length}-dim embedding`);
      successCount++;

      // Rate limiting: wait 1 second between requests
      if (i < phenotypes.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

    } catch (error: any) {
      console.error(`  ✗ Error: ${error.message}`);
      errorCount++;
    }

    console.log("");
  }

  console.log("\n" + "=".repeat(50));
  console.log("📊 EMBEDDING GENERATION COMPLETE");
  console.log("=".repeat(50));
  console.log(`✅ Successfully generated: ${successCount}`);
  console.log(`❌ Failed: ${errorCount}`);
  console.log(`📈 Total processed: ${phenotypes.length}`);
  console.log("");
}

main()
  .then(() => {
    console.log("✨ Done!");
    process.exit(0);
  })
  .catch((error) => {
    console.error("\n❌ Fatal error:", error);
    process.exit(1);
  });
