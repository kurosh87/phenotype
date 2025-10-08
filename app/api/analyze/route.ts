import { NextRequest, NextResponse } from "next/server";
import { stackServerApp } from "@/app/stack";
import { generateImageEmbedding } from "@/lib/embeddings";
import {
  findSimilarPhenotypes,
  saveUserUpload,
  saveAnalysisHistory,
} from "@/lib/database";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const user = await stackServerApp.getUser();
    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const { imageUrl } = body;

    if (!imageUrl) {
      return NextResponse.json(
        { error: "Image URL is required" },
        { status: 400 }
      );
    }

    // Step 1: Generate embedding for uploaded image
    console.log("Generating embedding for uploaded image...");
    const embedding = await generateImageEmbedding(imageUrl);

    // Step 2: Find similar phenotypes
    console.log("Finding similar phenotypes...");
    const matches = await findSimilarPhenotypes(embedding, 10);

    if (matches.length === 0) {
      return NextResponse.json(
        { error: "No phenotypes found in database. Please contact admin." },
        { status: 404 }
      );
    }

    // Step 3: Save user upload
    console.log("Saving user upload...");
    const uploadId = await saveUserUpload(user.id, imageUrl, embedding);

    // Step 4: Generate AI report using top matches
    console.log("Generating AI report...");
    const topMatches = matches.slice(0, 5);
    const aiReport = await generateAnalysisReport(topMatches);

    // Step 5: Save analysis history
    console.log("Saving analysis history...");
    const analysisId = await saveAnalysisHistory(
      user.id,
      uploadId,
      matches,
      aiReport
    );

    return NextResponse.json({
      analysisId,
      matches,
      aiReport,
      uploadedImageUrl: imageUrl,
    });
  } catch (error: any) {
    console.error("Analysis error:", error);
    return NextResponse.json(
      {
        error: "Failed to analyze image",
        details: error.message || "Unknown error",
      },
      { status: 500 }
    );
  }
}

async function generateAnalysisReport(matches: any[]): Promise<string> {
  const matchDescriptions = matches
    .map(
      (m, i) =>
        `${i + 1}. ${m.name} (${Math.round(m.similarity * 100)}% match)${
          m.geographicOrigin ? ` - Origin: ${m.geographicOrigin}` : ""
        }${m.description ? `\n   ${m.description}` : ""}`
    )
    .join("\n\n");

  const prompt = `You are an anthropological phenotype analyst. A user has uploaded a facial photo, and our system has matched it against a reference database of anthropological phenotypes.

Here are the top 5 matches:

${matchDescriptions}

Please write a detailed, educational report (200-300 words) that:
1. Summarizes the key phenotypic characteristics suggested by these matches
2. Explains what these matches might indicate about facial structure and features
3. Discusses the geographic and anthropological context if relevant
4. Is respectful, scientific, and educational in tone
5. Includes a disclaimer that this is computational analysis for educational purposes

Write the report in a professional yet accessible style.`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
      max_tokens: 500,
    });

    return (
      completion.choices[0]?.message?.content ||
      "Unable to generate analysis report at this time."
    );
  } catch (error) {
    console.error("OpenAI API error:", error);
    return `Analysis of your photo shows strong matches with the following phenotypes: ${matches
      .slice(0, 3)
      .map((m) => m.name)
      .join(", ")}. This suggests certain facial characteristics and structural features. Note: This is a computational analysis for educational purposes only.`;
  }
}
