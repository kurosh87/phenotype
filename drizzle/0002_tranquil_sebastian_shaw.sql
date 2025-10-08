CREATE TABLE "similar_phenotypes" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"phenotype_id" uuid NOT NULL,
	"similar_phenotype_id" uuid NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "phenotypes" ADD COLUMN "phenotype_groups" jsonb;--> statement-breakpoint
ALTER TABLE "phenotypes" ADD COLUMN "physical_traits" text;--> statement-breakpoint
ALTER TABLE "phenotypes" ADD COLUMN "literature" jsonb;--> statement-breakpoint
ALTER TABLE "phenotypes" ADD COLUMN "male_image_url" text;--> statement-breakpoint
ALTER TABLE "phenotypes" ADD COLUMN "female_image_url" text;--> statement-breakpoint
ALTER TABLE "phenotypes" ADD COLUMN "map_image_url" text;--> statement-breakpoint
ALTER TABLE "similar_phenotypes" ADD CONSTRAINT "similar_phenotypes_phenotype_id_phenotypes_id_fk" FOREIGN KEY ("phenotype_id") REFERENCES "public"."phenotypes"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "similar_phenotypes" ADD CONSTRAINT "similar_phenotypes_similar_phenotype_id_phenotypes_id_fk" FOREIGN KEY ("similar_phenotype_id") REFERENCES "public"."phenotypes"("id") ON DELETE cascade ON UPDATE no action;