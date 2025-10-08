import { redirect } from "next/navigation";
import { stackServerApp } from "@/app/stack";
import { getUserAnalysisHistory } from "@/lib/database";
import { ModernHeader } from "@/components/modern-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { Upload, History, TrendingUp, Sparkles, ArrowRight } from "lucide-react";

export default async function DashboardPage() {
  const user = await stackServerApp.getUser();

  if (!user) {
    redirect("/handler/sign-in");
  }

  const history = await getUserAnalysisHistory(user.id, 20);

  const thisMonthCount = history.filter((h: any) => {
    const date = new Date(h.createdAt);
    const now = new Date();
    return (
      date.getMonth() === now.getMonth() &&
      date.getFullYear() === now.getFullYear()
    );
  }).length;

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
            <div className="flex items-center gap-3 mb-3">
              <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
                  Your Dashboard
                </h1>
                <p className="text-muted-foreground text-lg mt-1">
                  Welcome back, {user.displayName || user.primaryEmail?.split("@")[0]}
                </p>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-12 animate-slide-up">
            <Card className="hover:shadow-xl transition-all duration-300 hover:-translate-y-1 border-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Total Analyses
                </CardTitle>
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <History className="h-5 w-5 text-primary" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold mb-1">{history.length}</div>
                <p className="text-xs text-muted-foreground">
                  Photos analyzed to date
                </p>
              </CardContent>
            </Card>

            <Card className="hover:shadow-xl transition-all duration-300 hover:-translate-y-1 border-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  This Month
                </CardTitle>
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <TrendingUp className="h-5 w-5 text-primary" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold mb-1">{thisMonthCount}</div>
                <p className="text-xs text-muted-foreground">
                  Analyses this month
                </p>
              </CardContent>
            </Card>

            <Card className="hover:shadow-xl transition-all duration-300 hover:-translate-y-1 border-2 bg-primary/5">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Quick Actions
                </CardTitle>
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <Upload className="h-5 w-5 text-primary" />
                </div>
              </CardHeader>
              <CardContent>
                <Link href="/">
                  <Button className="w-full h-11">
                    <Upload className="w-4 h-4 mr-2" />
                    New Analysis
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </div>

          {/* Analysis History */}
          <Card className="border-2 shadow-lg">
            <CardHeader className="border-b bg-muted/30">
              <div className="flex items-center justify-between">
                <CardTitle className="text-2xl">Recent Analyses</CardTitle>
                {history.length > 0 && (
                  <Badge variant="outline" className="text-sm px-3 py-1">
                    {history.length} total
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              {history.length === 0 ? (
                <div className="text-center py-16">
                  <div className="flex justify-center mb-6">
                    <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center">
                      <Upload className="w-10 h-10 text-primary" />
                    </div>
                  </div>
                  <h3 className="text-2xl font-bold mb-3">
                    No analyses yet
                  </h3>
                  <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
                    Upload your first photo to discover your phenotype matches
                  </p>
                  <Link href="/">
                    <Button size="lg" className="h-12 px-8">
                      <Upload className="w-4 h-4 mr-2" />
                      Upload Photo
                    </Button>
                  </Link>
                </div>
              ) : (
                <div className="space-y-4">
                  {history.map((item: any) => {
                    const matches = JSON.parse(item.topMatches || "[]");
                    const topMatch = matches[0];

                    return (
                      <Link
                        key={item.id}
                        href={`/analysis/${item.id}`}
                        className="block group"
                      >
                        <Card className="hover:shadow-xl transition-all duration-300 hover:-translate-y-1 border-2 hover:border-primary/50">
                          <CardContent className="p-6">
                            <div className="flex items-center gap-6">
                              {/* Uploaded Image Thumbnail */}
                              <div className="flex-shrink-0">
                                <div className="relative">
                                  <img
                                    src={item.uploadImageUrl}
                                    alt="Analysis"
                                    className="w-24 h-24 rounded-xl object-cover ring-2 ring-border"
                                  />
                                </div>
                              </div>

                              {/* Analysis Info */}
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between mb-3">
                                  <h3 className="font-bold text-lg">
                                    {new Date(item.createdAt).toLocaleDateString("en-US", {
                                      month: "long",
                                      day: "numeric",
                                      year: "numeric",
                                    })}
                                  </h3>
                                  <Badge variant="outline" className="ml-2">
                                    {matches.length} matches
                                  </Badge>
                                </div>

                                {topMatch && (
                                  <div className="flex items-center gap-3 mb-3">
                                    <span className="text-sm text-muted-foreground">
                                      Best match:
                                    </span>
                                    <span className="font-semibold text-base">
                                      {topMatch.phenotypeName}
                                    </span>
                                    <Badge className="bg-primary">
                                      {Math.round(topMatch.similarity * 100)}%
                                    </Badge>
                                  </div>
                                )}

                                <p className="text-sm text-muted-foreground line-clamp-2">
                                  {item.aiReport?.substring(0, 200)}...
                                </p>
                              </div>

                              {/* Top Match Thumbnail */}
                              {topMatch && (
                                <div className="flex-shrink-0 hidden sm:flex items-center gap-4">
                                  <img
                                    src={topMatch.imageUrl}
                                    alt={topMatch.phenotypeName}
                                    className="w-20 h-20 rounded-xl object-cover ring-2 ring-border"
                                  />
                                  <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      </Link>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          {/* CTA Section */}
          {history.length > 0 && (
            <div className="mt-12 text-center animate-fade-in">
              <Link href="/phenotypes">
                <Card className="p-8 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 cursor-pointer border-2 hover:border-primary/50 inline-block">
                  <div className="flex items-center space-x-4">
                    <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                      <Sparkles className="h-6 w-6 text-primary" />
                    </div>
                    <div className="text-left">
                      <h3 className="text-xl font-bold mb-1">Explore Phenotypes</h3>
                      <p className="text-muted-foreground text-sm">
                        Browse the complete gallery of reference phenotypes
                      </p>
                    </div>
                    <ArrowRight className="h-6 w-6 text-muted-foreground" />
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
