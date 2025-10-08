import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import * as schema from "@/app/schema/schema";
import { sql } from "drizzle-orm";

// Create database connection
const connection = neon(process.env.DATABASE_URL!);
export const db = drizzle(connection, { schema });

export interface PhenotypeMatch {
  id: string;
  name: string;
  description: string | null;
  geographicOrigin: string | null;
  imageUrl: string;
  similarity: number;
}

/**
 * Find similar phenotypes using vector similarity search
 * Uses pgvector's cosine distance operator (<=>)
 */
export async function findSimilarPhenotypes(
  embedding: number[],
  limit: number = 10
): Promise<PhenotypeMatch[]> {
  try {
    // Convert embedding array to proper format for pgvector
    const embeddingString = `[${embedding.join(",")}]`;

    const results = await connection`
      SELECT
        id,
        name,
        description,
        geographic_origin as "geographicOrigin",
        image_url as "imageUrl",
        1 - (embedding <=> ${embeddingString}::vector) as similarity
      FROM phenotypes
      WHERE embedding IS NOT NULL
      ORDER BY embedding <=> ${embeddingString}::vector
      LIMIT ${limit}
    `;

    return results.map((row: any) => ({
      id: row.id,
      name: row.name,
      description: row.description,
      geographicOrigin: row.geographicOrigin,
      imageUrl: row.imageUrl,
      similarity: parseFloat(row.similarity),
    }));
  } catch (error) {
    console.error("Error finding similar phenotypes:", error);
    throw new Error(`Database query failed: ${error}`);
  }
}

/**
 * Save user upload with embedding
 */
export async function saveUserUpload(
  userId: string,
  imageUrl: string,
  embedding: number[]
) {
  const embeddingString = `[${embedding.join(",")}]`;

  const result = await connection`
    INSERT INTO user_uploads (user_id, image_url, embedding, created_at)
    VALUES (${userId}, ${imageUrl}, ${embeddingString}::vector, NOW())
    RETURNING id
  `;

  return result[0].id;
}

/**
 * Save analysis history
 */
export async function saveAnalysisHistory(
  userId: string,
  uploadId: string,
  topMatches: PhenotypeMatch[],
  aiReport: string
) {
  const matchesJson = JSON.stringify(
    topMatches.map((m) => ({
      phenotypeId: m.id,
      phenotypeName: m.name,
      similarity: m.similarity,
      imageUrl: m.imageUrl,
    }))
  );

  const result = await connection`
    INSERT INTO analysis_history (user_id, upload_id, top_matches, ai_report, created_at)
    VALUES (${userId}, ${uploadId}, ${matchesJson}::jsonb, ${aiReport}, NOW())
    RETURNING id
  `;

  return result[0].id;
}

/**
 * Get user's analysis history
 */
export async function getUserAnalysisHistory(userId: string, limit: number = 20) {
  return await connection`
    SELECT
      ah.id,
      ah.created_at as "createdAt",
      ah.top_matches as "topMatches",
      ah.ai_report as "aiReport",
      uu.image_url as "uploadImageUrl"
    FROM analysis_history ah
    JOIN user_uploads uu ON ah.upload_id = uu.id
    WHERE ah.user_id = ${userId}
    ORDER BY ah.created_at DESC
    LIMIT ${limit}
  `;
}

/**
 * Get specific analysis by ID
 */
export async function getAnalysisById(analysisId: string, userId: string) {
  const result = await connection`
    SELECT
      ah.id,
      ah.created_at as "createdAt",
      ah.top_matches as "topMatches",
      ah.ai_report as "aiReport",
      uu.image_url as "uploadImageUrl"
    FROM analysis_history ah
    JOIN user_uploads uu ON ah.upload_id = uu.id
    WHERE ah.id = ${analysisId} AND ah.user_id = ${userId}
  `;

  return result[0] || null;
}
