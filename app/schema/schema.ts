import { bigint, boolean, pgTable, text, timestamp, uuid, jsonb, vector } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

// Original todos table (kept for compatibility)
export const todos = pgTable("todos", {
  id: bigint("id", { mode: "bigint" })
    .primaryKey()
    .generatedByDefaultAsIdentity(),
  ownerId: text("owner_id").notNull(),
  task: text("task").notNull(),
  isComplete: boolean("is_complete").notNull().default(false),
  insertedAt: timestamp("inserted_at", { withTimezone: true })
    .defaultNow()
    .notNull(),
});

// Phenotype reference database table
export const phenotypes = pgTable("phenotypes", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: text("name").notNull(),

  // Phenotype can belong to multiple groups (e.g., "Mediterranid, Orientalid")
  phenotypeGroups: jsonb("phenotype_groups").$type<string[]>(),

  description: text("description"),
  physicalTraits: text("physical_traits"),
  geographicOrigin: text("geographic_origin"),
  keyTraits: jsonb("key_traits").$type<string[]>(),

  // Literature references
  literature: jsonb("literature").$type<Array<{text: string, url?: string}>>(),

  // Images
  maleImageUrl: text("male_image_url"),
  femaleImageUrl: text("female_image_url"),
  mapImageUrl: text("map_image_url"),
  imageUrl: text("image_url").notNull(), // Primary/composite image

  // Vector embedding for similarity search
  embedding: vector("embedding", { dimensions: 512 }), // CLIP ViT-B/32 embeddings are 512-dimensional

  // Geographic distribution (GeoJSON FeatureCollection)
  distributionGeojson: jsonb("distribution_geojson").$type<{
    type: 'FeatureCollection',
    features: Array<{
      type: 'Feature',
      geometry: { type: 'Polygon', coordinates: number[][][] },
      properties: { intensity: 'primary' | 'secondary', color: string, area_pixels: number }
    }>
  }>(),

  // Additional metadata
  metadata: jsonb("metadata").$type<Record<string, any>>(),

  createdAt: timestamp("created_at", { withTimezone: true })
    .defaultNow()
    .notNull(),
});

// Similar phenotypes relationship table (many-to-many)
export const similarPhenotypes = pgTable("similar_phenotypes", {
  id: uuid("id").primaryKey().defaultRandom(),
  phenotypeId: uuid("phenotype_id")
    .notNull()
    .references(() => phenotypes.id, { onDelete: "cascade" }),
  similarPhenotypeId: uuid("similar_phenotype_id")
    .notNull()
    .references(() => phenotypes.id, { onDelete: "cascade" }),
  createdAt: timestamp("created_at", { withTimezone: true })
    .defaultNow()
    .notNull(),
});

// User uploaded photos table
export const userUploads = pgTable("user_uploads", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: text("user_id").notNull(), // References Stack Auth user
  imageUrl: text("image_url").notNull(), // Vercel Blob URL
  embedding: vector("embedding", { dimensions: 512 }),
  analysisResults: jsonb("analysis_results").$type<any>(),
  createdAt: timestamp("created_at", { withTimezone: true })
    .defaultNow()
    .notNull(),
});

// Analysis history table
export const analysisHistory = pgTable("analysis_history", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: text("user_id").notNull(),
  uploadId: uuid("upload_id")
    .notNull()
    .references(() => userUploads.id, { onDelete: "cascade" }),
  topMatches: jsonb("top_matches").$type<Array<{
    phenotypeId: string;
    phenotypeName: string;
    similarity: number;
    imageUrl: string;
  }>>(),
  aiReport: text("ai_report"),
  createdAt: timestamp("created_at", { withTimezone: true })
    .defaultNow()
    .notNull(),
});
