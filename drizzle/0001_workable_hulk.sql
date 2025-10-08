CREATE TABLE "analysis_history" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" text NOT NULL,
	"upload_id" uuid NOT NULL,
	"top_matches" jsonb,
	"ai_report" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "phenotypes" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" text NOT NULL,
	"description" text,
	"geographic_origin" text,
	"key_traits" jsonb,
	"image_url" text NOT NULL,
	"embedding" vector(512),
	"metadata" jsonb,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "user_uploads" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" text NOT NULL,
	"image_url" text NOT NULL,
	"embedding" vector(512),
	"analysis_results" jsonb,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "analysis_history" ADD CONSTRAINT "analysis_history_upload_id_user_uploads_id_fk" FOREIGN KEY ("upload_id") REFERENCES "public"."user_uploads"("id") ON DELETE cascade ON UPDATE no action;