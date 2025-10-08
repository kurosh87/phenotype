#!/usr/bin/env tsx
/**
 * Deduplicate Phenotypes
 * Removes duplicate phenotype entries, keeping the most recent one
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { neon } from "@neondatabase/serverless";

const connection = neon(process.env.DATABASE_URL!);

async function deduplicatePhenotypes() {
  console.log("🔍 Finding duplicates...\n");

  // Find duplicates
  const duplicates = await connection`
    SELECT name, COUNT(*) as count, ARRAY_AGG(id ORDER BY created_at DESC) as ids
    FROM phenotypes
    GROUP BY name
    HAVING COUNT(*) > 1
    ORDER BY count DESC, name
  `;

  if (duplicates.length === 0) {
    console.log("✅ No duplicates found!");
    return;
  }

  console.log(`Found ${duplicates.length} duplicate names:\n`);

  for (const dup of duplicates) {
    console.log(`  - ${dup.name}: ${dup.count} copies`);
  }

  console.log(`\n🗑️  Removing duplicates (keeping most recent)...\n`);

  let totalDeleted = 0;

  for (const dup of duplicates) {
    const idsToKeep = [dup.ids[0]]; // Keep first (most recent)
    const idsToDelete = dup.ids.slice(1); // Delete rest

    console.log(`Processing "${dup.name}": keeping 1, deleting ${idsToDelete.length}`);

    // Delete similarity relationships first (foreign key constraint)
    await connection`
      DELETE FROM similar_phenotypes
      WHERE phenotype_id = ANY(${idsToDelete})
         OR similar_phenotype_id = ANY(${idsToDelete})
    `;

    // Delete the duplicate phenotypes
    const result = await connection`
      DELETE FROM phenotypes
      WHERE id = ANY(${idsToDelete})
    `;

    totalDeleted += idsToDelete.length;
  }

  console.log(`\n✅ Deduplication complete!`);
  console.log(`   Deleted: ${totalDeleted} duplicate records`);

  // Show final stats
  const stats = await connection`
    SELECT
      (SELECT COUNT(*) FROM phenotypes) as phenotypes,
      (SELECT COUNT(*) FROM similar_phenotypes) as relationships
  `;

  console.log(`\n📊 Final Stats:`);
  console.log(`   Phenotypes: ${stats[0].phenotypes}`);
  console.log(`   Relationships: ${stats[0].relationships}`);
}

deduplicatePhenotypes().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
