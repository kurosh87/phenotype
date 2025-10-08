#!/usr/bin/env tsx
/**
 * Fix Basic Phenotype Groups
 * The 38 basic phenotypes should have themselves as their own group
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { neon } from "@neondatabase/serverless";
import fs from "fs";

const connection = neon(process.env.DATABASE_URL!);

// The 38 basic phenotypes from http://humanphenotypes.net/basic/
const BASIC_PHENOTYPES = [
  "Ainuid", "Alpinid", "Amazonid", "Andid", "Armenoid", "Australid",
  "Bambutid", "Bantuid", "Centralid", "Congolid", "Dinarid", "East Europid",
  "Eskimid", "Ethiopid", "Indid", "Indo Melanid", "Khoid", "Lagid",
  "Lappid", "Margid", "Mediterranid", "Melanesid", "Negritid", "Nilotid",
  "Nordid", "Orientalid", "Pacifid", "Patagonid", "Polynesid", "Sanid",
  "Sibirid", "Silvid", "Sinid", "South Mongolid", "Sudanid", "Tungid",
  "Turanid", "Veddid"
];

async function fixBasicGroups() {
  console.log("🔧 Fixing basic phenotype groups...\n");

  let fixed = 0;

  for (const name of BASIC_PHENOTYPES) {
    // Check if exists and has no groups
    const result = await connection`
      SELECT id, name, phenotype_groups
      FROM phenotypes
      WHERE name = ${name}
    `;

    if (result.length === 0) {
      console.log(`  ⚠️  Not found: ${name}`);
      continue;
    }

    const phenotype = result[0];
    const currentGroups = phenotype.phenotype_groups || [];

    if (currentGroups.length === 0) {
      // Update to have itself as the group
      await connection`
        UPDATE phenotypes
        SET phenotype_groups = ${JSON.stringify([name])}::jsonb
        WHERE id = ${phenotype.id}
      `;
      console.log(`  ✓ Fixed: ${name} -> [${name}]`);
      fixed++;
    } else {
      console.log(`  ○ Already has groups: ${name} -> ${JSON.stringify(currentGroups)}`);
    }
  }

  console.log(`\n✅ Fixed ${fixed} basic phenotypes`);

  // Show final stats
  const stats = await connection`
    SELECT
      (SELECT COUNT(*) FROM phenotypes WHERE phenotype_groups IS NULL OR jsonb_array_length(phenotype_groups) = 0) as no_groups,
      (SELECT COUNT(*) FROM phenotypes WHERE jsonb_array_length(phenotype_groups) = 1) as single_group,
      (SELECT COUNT(*) FROM phenotypes WHERE jsonb_array_length(phenotype_groups) > 1) as multi_groups
  `;

  console.log(`\n📊 Final Group Stats:`);
  console.log(`  No groups: ${stats[0].no_groups}`);
  console.log(`  Single group: ${stats[0].single_group}`);
  console.log(`  Multiple groups: ${stats[0].multi_groups}`);
}

fixBasicGroups().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
