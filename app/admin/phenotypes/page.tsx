import { redirect } from "next/navigation";
import { stackServerApp } from "@/app/stack";
import { AdminPhenotypeForm } from "@/components/admin-phenotype-form";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

async function getPhenotypes() {
  try {
    const response = await fetch(
      `${process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3002"}/api/admin/phenotypes`,
      {
        cache: "no-store",
      }
    );
    if (!response.ok) return [];
    const data = await response.json();
    return data.phenotypes || [];
  } catch (error) {
    console.error("Error fetching phenotypes:", error);
    return [];
  }
}

export default async function AdminPhenotypesPage() {
  const user = await stackServerApp.getUser();

  if (!user) {
    redirect("/handler/sign-in");
  }

  // TODO: Add proper admin check in production
  const phenotypes = await getPhenotypes();

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link href="/">
            <Button variant="ghost" className="mb-4">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
          </Link>
          <h1 className="text-4xl font-bold">Admin: Manage Phenotypes</h1>
          <p className="text-muted-foreground mt-2">
            Add and manage reference phenotypes for matching
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - Add Form */}
          <div>
            <AdminPhenotypeForm />
          </div>

          {/* Right Column - Existing Phenotypes */}
          <div>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Existing Phenotypes</CardTitle>
                  <Badge variant="outline">{phenotypes.length} total</Badge>
                </div>
              </CardHeader>
              <CardContent>
                {phenotypes.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <p>No phenotypes added yet.</p>
                    <p className="text-sm mt-2">
                      Add your first phenotype using the form on the left.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4 max-h-[600px] overflow-y-auto">
                    {phenotypes.map((phenotype: any) => (
                      <Card key={phenotype.id}>
                        <CardContent className="pt-6">
                          <div className="flex gap-4">
                            <img
                              src={phenotype.imageUrl}
                              alt={phenotype.name}
                              className="w-20 h-20 rounded-lg object-cover"
                            />
                            <div className="flex-1 min-w-0">
                              <h3 className="font-semibold">
                                {phenotype.name}
                              </h3>
                              {phenotype.geographicOrigin && (
                                <p className="text-sm text-muted-foreground">
                                  {phenotype.geographicOrigin}
                                </p>
                              )}
                              {phenotype.description && (
                                <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                  {phenotype.description}
                                </p>
                              )}
                              <p className="text-xs text-muted-foreground mt-2">
                                Added{" "}
                                {new Date(
                                  phenotype.createdAt
                                ).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
