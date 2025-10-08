import { redirect } from "next/navigation";
import { stackServerApp } from "@/app/stack";
import { ModernHeader } from "@/components/modern-header";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowLeft, MapPin } from "lucide-react";
import { neon } from "@neondatabase/serverless";
import PhenotypeDistributionMap from "@/components/PhenotypeDistributionMap";
import Image from "next/image";

interface PhenotypeDetail {
  id: number;
  name: string;
  description: string | null;
  geographicOrigin: string | null;
  keyTraits: string[] | null;
  physicalTraits: string | null;
  imageUrl: string;
  maleImageUrl: string | null;
  femaleImageUrl: string | null;
  mapImageUrl: string | null;
  distributionGeojson: any;
  phenotypeGroups: string[] | null;
}

async function getPhenotype(id: string): Promise<PhenotypeDetail | null> {
  try {
    const connection = neon(process.env.DATABASE_URL!);
    const result = await connection`
      SELECT
        id,
        name,
        description,
        geographic_origin as "geographicOrigin",
        key_traits as "keyTraits",
        physical_traits as "physicalTraits",
        image_url as "imageUrl",
        male_image_url as "maleImageUrl",
        female_image_url as "femaleImageUrl",
        map_image_url as "mapImageUrl",
        distribution_geojson as "distributionGeojson",
        phenotype_groups as "phenotypeGroups"
      FROM phenotypes
      WHERE id = ${id}
      LIMIT 1
    `;
    return result[0] || null;
  } catch (error) {
    console.error("Error fetching phenotype:", error);
    return null;
  }
}

export default async function PhenotypeDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const user = await stackServerApp.getUser();

  if (!user) {
    redirect("/handler/sign-in");
  }

  const phenotype = await getPhenotype(params.id);

  if (!phenotype) {
    redirect("/phenotypes");
  }

  const userData = {
    displayName: user.displayName,
    primaryEmail: user.primaryEmail,
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <ModernHeader user={userData} />

      <main className="flex-1 container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-6">
          <Link href="/phenotypes">
            <Button variant="ghost" size="sm">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Phenotypes
            </Button>
          </Link>
        </div>

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            {phenotype.name}
          </h1>
          {phenotype.geographicOrigin && (
            <div className="flex items-center text-gray-600">
              <MapPin className="h-4 w-4 mr-1" />
              <span>{phenotype.geographicOrigin}</span>
            </div>
          )}
        </div>

        {/* Groups */}
        {phenotype.phenotypeGroups && phenotype.phenotypeGroups.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Groups:
            </h3>
            <div className="flex flex-wrap gap-2">
              {phenotype.phenotypeGroups.map((group) => (
                <Badge key={group} variant="secondary">
                  {group}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Description */}
        {phenotype.description && (
          <Card className="mb-8">
            <CardContent className="pt-6">
              <h2 className="text-xl font-semibold mb-3">Description</h2>
              <p className="text-gray-700 leading-relaxed">
                {phenotype.description}
              </p>
            </CardContent>
          </Card>
        )}

        {/* Side-by-side Map Comparison */}
        <Card className="mb-8">
          <CardContent className="pt-6">
            <h2 className="text-xl font-semibold mb-4">
              Geographic Distribution - Map Comparison
            </h2>
            <p className="text-sm text-gray-600 mb-4">
              Compare the original map (left) with the extracted interactive map
              (right) for accuracy verification
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Original Map Image */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-gray-700">
                  Original Map Image
                </h3>
                {phenotype.mapImageUrl ? (
                  <div className="relative bg-gray-100 rounded-lg overflow-hidden border border-gray-200 aspect-[651/332]">
                    <Image
                      src={phenotype.mapImageUrl}
                      alt={`${phenotype.name} distribution map`}
                      fill
                      className="object-contain"
                    />
                  </div>
                ) : (
                  <div className="flex items-center justify-center bg-gray-100 rounded-lg border border-gray-200 aspect-[651/332]">
                    <p className="text-gray-500 text-sm">
                      No map image available
                    </p>
                  </div>
                )}
              </div>

              {/* Interactive Mapbox Map */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-gray-700">
                  Extracted Interactive Map
                </h3>
                <PhenotypeDistributionMap
                  geojson={phenotype.distributionGeojson}
                  phenotypeName={phenotype.name}
                  className="h-[400px] border border-gray-200"
                />
              </div>
            </div>

            {/* Legend for colors */}
            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-sm text-blue-900">
                <strong>Color Legend:</strong> Bright yellow represents{" "}
                <strong>primary distribution</strong> areas where this phenotype
                is most commonly found. Olive/dark yellow indicates{" "}
                <strong>secondary distribution</strong> areas with lower
                frequency.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Composite Images */}
        {(phenotype.maleImageUrl || phenotype.femaleImageUrl) && (
          <Card className="mb-8">
            <CardContent className="pt-6">
              <h2 className="text-xl font-semibold mb-4">
                Representative Composites
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {phenotype.maleImageUrl && (
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-gray-700">Male</h3>
                    <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                      <Image
                        src={phenotype.maleImageUrl}
                        alt={`${phenotype.name} male composite`}
                        fill
                        className="object-cover"
                      />
                    </div>
                  </div>
                )}
                {phenotype.femaleImageUrl && (
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-gray-700">
                      Female
                    </h3>
                    <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                      <Image
                        src={phenotype.femaleImageUrl}
                        alt={`${phenotype.name} female composite`}
                        fill
                        className="object-cover"
                      />
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Physical Traits */}
        {phenotype.physicalTraits && (
          <Card className="mb-8">
            <CardContent className="pt-6">
              <h2 className="text-xl font-semibold mb-3">Physical Traits</h2>
              <p className="text-gray-700 leading-relaxed whitespace-pre-line">
                {phenotype.physicalTraits}
              </p>
            </CardContent>
          </Card>
        )}

        {/* Key Traits */}
        {phenotype.keyTraits && phenotype.keyTraits.length > 0 && (
          <Card>
            <CardContent className="pt-6">
              <h2 className="text-xl font-semibold mb-3">Key Traits</h2>
              <ul className="list-disc list-inside space-y-1 text-gray-700">
                {phenotype.keyTraits.map((trait, index) => (
                  <li key={index}>{trait}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}
