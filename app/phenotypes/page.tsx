import { redirect } from "next/navigation";
import { stackServerApp } from "@/app/stack";
import { ModernHeader } from "@/components/modern-header";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowLeft, MapPin, Sparkles, Search } from "lucide-react";
import { neon } from "@neondatabase/serverless";

async function getAllPhenotypes() {
  try {
    const connection = neon(process.env.DATABASE_URL!);
    const phenotypes = await connection`
      SELECT
        id,
        name,
        description,
        geographic_origin as "geographicOrigin",
        key_traits as "keyTraits",
        image_url as "imageUrl",
        created_at as "createdAt"
      FROM phenotypes
      ORDER BY name ASC
    `;
    return phenotypes;
  } catch (error) {
    console.error("Error fetching phenotypes:", error);
    return [];
  }
}

export default async function PhenotypesPage() {
  const user = await stackServerApp.getUser();

  if (!user) {
    redirect("/handler/sign-in");
  }

  const phenotypes = await getAllPhenotypes();

  const userData = {
    displayName: user.displayName,
    primaryEmail: user.primaryEmail,
  };

  return (
    <div className="min-h-screen flex flex-col">
      <ModernHeader user={userData} />

      <main className="flex-1 gradient-mesh">
        <div className="container mx-auto px-4 py-12 md:py-16">
          {/* Header */}
          <div className="mb-12 animate-fade-in">
            <Link href="/">
              <Button variant="ghost" className="mb-6 hover:bg-primary/10">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Home
              </Button>
            </Link>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
              <div className="flex items-center gap-3">
                <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                  <Sparkles className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
                    Phenotype Gallery
                  </h1>
                  <p className="text-muted-foreground text-lg mt-1">
                    Browse all reference phenotypes in our database
                  </p>
                </div>
              </div>
              <Badge variant="outline" className="text-lg px-6 py-3 w-fit">
                <Search className="w-4 h-4 mr-2" />
                {phenotypes.length} phenotypes
              </Badge>
            </div>
          </div>

          {phenotypes.length === 0 ? (
            <Card className="border-2 shadow-lg">
              <CardContent className="py-16 text-center">
                <div className="flex justify-center mb-6">
                  <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center">
                    <Sparkles className="w-10 h-10 text-primary" />
                  </div>
                </div>
                <h3 className="text-2xl font-bold mb-3">
                  No phenotypes yet
                </h3>
                <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
                  The database is being populated. Check back soon!
                </p>
                <Link href="/admin/phenotypes">
                  <Button size="lg" className="h-12 px-8">
                    Add Phenotypes
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-slide-up">
              {phenotypes.map((phenotype: any) => (
                <Link key={phenotype.id} href={`/phenotypes/${phenotype.id}`}>
                  <Card className="overflow-hidden hover:shadow-xl transition-all duration-300 hover:-translate-y-2 border-2 group cursor-pointer">
                    <div className="aspect-square relative overflow-hidden">
                      <img
                        src={phenotype.imageUrl}
                        alt={phenotype.name}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    </div>
                    <CardContent className="p-5">
                    <h3 className="font-bold text-xl mb-3 group-hover:text-primary transition-colors">
                      {phenotype.name}
                    </h3>

                    {phenotype.geographicOrigin && (
                      <div className="mb-3">
                        <Badge variant="secondary" className="text-xs font-medium">
                          <MapPin className="w-3 h-3 mr-1" />
                          {phenotype.geographicOrigin}
                        </Badge>
                      </div>
                    )}

                    {phenotype.description && (
                      <p className="text-sm text-muted-foreground line-clamp-3 mb-3 leading-relaxed">
                        {phenotype.description}
                      </p>
                    )}

                    {phenotype.keyTraits && (
                      <div className="flex flex-wrap gap-1.5">
                        {(Array.isArray(phenotype.keyTraits)
                          ? phenotype.keyTraits
                          : []
                        )
                          .slice(0, 3)
                          .map((trait: string, index: number) => (
                            <Badge
                              key={index}
                              variant="outline"
                              className="text-xs font-normal"
                            >
                              {trait}
                            </Badge>
                          ))}
                        {phenotype.keyTraits.length > 3 && (
                          <Badge
                            variant="outline"
                            className="text-xs font-normal"
                          >
                            +{phenotype.keyTraits.length - 3}
                          </Badge>
                        )}
                      </div>
                    )}
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          )}

          {/* Bottom CTA */}
          {phenotypes.length > 0 && (
            <div className="mt-12 text-center animate-fade-in">
              <Link href="/">
                <Card className="p-8 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 cursor-pointer border-2 hover:border-primary/50 inline-block">
                  <div className="flex items-center space-x-4">
                    <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                      <Sparkles className="h-6 w-6 text-primary" />
                    </div>
                    <div className="text-left">
                      <h3 className="text-xl font-bold mb-1">Try Your Own Photo</h3>
                      <p className="text-muted-foreground text-sm">
                        Upload a photo to find your phenotype matches
                      </p>
                    </div>
                  </div>
                </Card>
              </Link>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
