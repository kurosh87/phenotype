import Replicate from "replicate";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN!,
});

/**
 * Generate image embeddings using CLIP model from Replicate
 * Returns a 512-dimensional vector embedding
 */
export async function generateImageEmbedding(imageUrl: string): Promise<number[]> {
  try {
    const output = await replicate.run(
      "andreasjansson/clip-features:75b33f253f7714a281ad3e9b28f63e3232d583716ef6718f2e46641077ea040a",
      {
        input: {
          inputs: imageUrl,
        },
      }
    ) as any;

    // The CLIP model returns: [{embedding: [...], input: ReadableStream}]
    let embedding: number[] | undefined;

    if (Array.isArray(output) && output.length > 0) {
      // Extract embedding from first result
      const firstResult = output[0];
      if (firstResult && firstResult.embedding) {
        embedding = firstResult.embedding;
      } else if (Array.isArray(firstResult)) {
        // Sometimes it's just nested array
        embedding = firstResult;
      }
    } else if (output && output.embedding) {
      // Sometimes it's directly an object with embedding
      embedding = output.embedding;
    }

    if (!Array.isArray(embedding) || embedding.length === 0) {
      console.error("Unexpected output format:", JSON.stringify(output).substring(0, 200));
      throw new Error("Invalid embedding format received from CLIP model");
    }

    return embedding;
  } catch (error) {
    console.error("Error generating embedding:", error);
    throw new Error(`Failed to generate image embedding: ${error}`);
  }
}

/**
 * Calculate cosine similarity between two vectors
 * Returns a value between -1 and 1, where 1 means identical
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error("Vectors must have the same length");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (normA * normB);
}

/**
 * Format similarity score as percentage
 */
export function similarityToPercentage(similarity: number): number {
  return Math.round(similarity * 100);
}
