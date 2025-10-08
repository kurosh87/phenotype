#!/usr/bin/env ts-node
/**
 * Standalone script to upload GeoJSON distribution polygons to Neon database
 * Set DATABASE_URL environment variable before running
 */

import * as fs from "fs";
import * as path from "path";
import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import * as schema from "../app/schema/schema";
import { eq } from "drizzle-orm";

if (!process.env.DATABASE_URL) {
  console.error("Error: DATABASE_URL environment variable is not set");
  console.error("Usage: DATABASE_URL='...' npx tsx upload_geojson_standalone.ts <geojson_dir>");
  process.exit(1);
}

const connection = neon(process.env.DATABASE_URL);
const db = drizzle(connection, { schema });

interface GeoJSONFeature {
  type: "Feature";
  geometry: { type: "Polygon"; coordinates: number[][][] };
  properties: { intensity: "primary" | "secondary"; color: string; area_pixels: number };
}

interface GeoJSONFeatureCollection {
  type: "FeatureCollection";
  features: GeoJSONFeature[];
}

async function uploadGeoJSON(geojsonDir: string) {
  console.log(`📍 Uploading GeoJSON files from: ${geojsonDir}\n`);

  // Read all GeoJSON files
  const files = fs.readdirSync(geojsonDir).filter((f) => f.endsWith(".geojson"));

  console.log(`Found ${files.length} GeoJSON files\n`);

  let updated = 0;
  let notFound = 0;
  let errors = 0;
  const notFoundPhenotypes: string[] = [];

  for (const file of files) {
    const phenotypeName = file.replace(".geojson", "");
    const filePath = path.join(geojsonDir, file);

    try {
      // Read GeoJSON
      const geojsonData = fs.readFileSync(filePath, "utf-8");
      const geojson: GeoJSONFeatureCollection = JSON.parse(geojsonData);

      // Find phenotype by name (handle spaces and underscores)
      const normalizedName = phenotypeName.replace(/_/g, " ");

      const result = await db
        .select({ id: schema.phenotypes.id, name: schema.phenotypes.name })
        .from(schema.phenotypes)
        .where(eq(schema.phenotypes.name, normalizedName))
        .limit(1);

      if (result.length === 0) {
        notFound++;
        notFoundPhenotypes.push(normalizedName);
        console.log(`⚠️  ${normalizedName}: Not found in database`);
        continue;
      }

      const phenotype = result[0];

      // Update phenotype with GeoJSON
      await db
        .update(schema.phenotypes)
        .set({ distributionGeojson: geojson as any })
        .where(eq(schema.phenotypes.id, phenotype.id));

      updated++;

      const primaryCount = geojson.features.filter(f => f.properties.intensity === "primary").length;
      const secondaryCount = geojson.features.filter(f => f.properties.intensity === "secondary").length;

      console.log(`✅ ${normalizedName}: ${primaryCount} primary, ${secondaryCount} secondary regions`);

    } catch (error) {
      errors++;
      console.error(`❌ ${phenotypeName}: Error - ${error}`);
    }
  }

  // Summary
  console.log("\n" + "=".repeat(60));
  console.log(`Upload complete!`);
  console.log(`Total files: ${files.length}`);
  console.log(`Updated: ${updated}`);
  console.log(`Not found in DB: ${notFound}`);
  console.log(`Errors: ${errors}`);

  if (notFoundPhenotypes.length > 0) {
    console.log(`\nPhenotypes not found in database (${notFoundPhenotypes.length}):`);
    notFoundPhenotypes.slice(0, 10).forEach((name) => {
      console.log(`  - ${name}`);
    });
    if (notFoundPhenotypes.length > 10) {
      console.log(`  ... and ${notFoundPhenotypes.length - 10} more`);
    }
  }
}

// Main execution
const geojsonDir = process.argv[2] || "geojson_output";

if (!fs.existsSync(geojsonDir)) {
  console.error(`Error: Directory not found: ${geojsonDir}`);
  process.exit(1);
}

uploadGeoJSON(geojsonDir)
  .then(() => {
    console.log("\n✨ Done!");
    process.exit(0);
  })
  .catch((error) => {
    console.error("\n❌ Fatal error:", error);
    process.exit(1);
  });
