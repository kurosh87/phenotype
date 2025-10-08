import { notFound, redirect } from "next/navigation";
import { stackServerApp } from "@/app/stack";
import { getAnalysisById } from "@/lib/database";
import { ModernHeader } from "@/components/modern-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowLeft, Download, Upload, Sparkles, TrendingUp } from "lucide-react";

export default async function AnalysisPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const user = await stackServerApp.getUser();

  if (!user) {
    redirect("/handler/sign-in");
  }

  const { id } = await params;
  const analysis = await getAnalysisById(id, user.id);

  if (!analysis) {
    notFound();
  }

  const topMatches = JSON.parse(analysis.topMatches || "[]");
  const topMatch = topMatches[0];

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
            <Link href="/dashboard">
              <Button variant="ghost" className="mb-6 hover:bg-primary/10">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Dashboard
              </Button>
            </Link>
            <div className="flex items-center gap-3 mb-3">
              <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
                  Analysis Results
                </h1>
                <p className="text-muted-foreground text-lg mt-1">
                  Completed on {new Date(analysis.createdAt).toLocaleDateString("en-US", {
                    month: "long",
                    day: "numeric",
                    year: "numeric",
                  })}
                </p>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 animate-slide-up">
            {/* Left Column - Uploaded Image */}
            <div className="space-y-6">
              <Card className="border-2 shadow-lg overflow-hidden">
                <CardHeader className="border-b bg-muted/30">
                  <CardTitle className="text-xl">Your Photo</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <img
                    src={analysis.uploadImageUrl}
                    alt="Uploaded photo"
                    className="w-full aspect-square object-cover"
                  />
                </CardContent>
              </Card>

              {/* Top Match Summary */}
              {topMatch && (
                <Card className="border-2 shadow-lg bg-gradient-to-br from-primary/5 to-background">
                  <CardHeader className="border-b bg-muted/30">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xl">Best Match</CardTitle>
                      <TrendingUp className="h-5 w-5 text-primary" />
                    </div>
                  </CardHeader>
                  <CardContent className="p-6">
                    <div className="flex flex-col sm:flex-row items-center gap-6">
                      <div className="flex-shrink-0">
                        <img
                          src={topMatch.imageUrl}
                          alt={topMatch.phenotypeName}
                          className="w-32 h-32 rounded-xl object-cover ring-4 ring-primary/20"
                        />
                      </div>
                      <div className="flex-1 text-center sm:text-left">
                        <h3 className="text-2xl font-bold mb-3">
                          {topMatch.phenotypeName}
                        </h3>
                        <Badge className="text-lg px-4 py-2 bg-primary">
                          {Math.round(topMatch.similarity * 100)}% Match
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Right Column - Analysis Report & Matches */}
            <div className="space-y-6">
              {/* AI Report */}
              <Card className="border-2 shadow-lg">
                <CardHeader className="border-b bg-muted/30">
                  <CardTitle className="text-xl">AI Analysis Report</CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="prose prose-sm max-w-none">
                    <p className="text-sm leading-relaxed whitespace-pre-wrap text-foreground">
                      {analysis.aiReport}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* All Matches */}
              <Card className="border-2 shadow-lg">
                <CardHeader className="border-b bg-muted/30">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-xl">Top Phenotype Matches</CardTitle>
                    <Badge variant="outline" className="px-3 py-1">
                      {topMatches.length} matches
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="space-y-5">
                    {topMatches.map((match: any, index: number) => (
                      <div key={match.phenotypeId}>
                        {index > 0 && <Separator className="my-5" />}
                        <div className="flex items-start gap-4 group">
                          <div className="flex-shrink-0 relative">
                            <div className="absolute -left-2 -top-2 h-6 w-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center z-10">
                              {index + 1}
                            </div>
                            <img
                              src={match.imageUrl}
                              alt={match.phenotypeName}
                              className="w-20 h-20 rounded-xl object-cover ring-2 ring-border group-hover:ring-primary transition-all"
                            />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-2">
                              <h4 className="font-bold text-base">
                                {match.phenotypeName}
                              </h4>
                              <Badge className="ml-2 bg-primary">
                                {Math.round(match.similarity * 100)}%
                              </Badge>
                            </div>
                            <div className="mt-3 bg-secondary/30 rounded-full h-3 overflow-hidden">
                              <div
                                className="bg-primary h-full rounded-full transition-all duration-500"
                                style={{
                                  width: `${match.similarity * 100}%`,
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Actions */}
              <Card className="border-2 shadow-lg bg-muted/30">
                <CardContent className="p-6">
                  <div className="flex flex-col sm:flex-row gap-3">
                    <Button variant="outline" className="flex-1 h-12">
                      <Download className="w-4 h-4 mr-2" />
                      Download Report
                    </Button>
                    <Link href="/" className="flex-1">
                      <Button className="w-full h-12">
                        <Upload className="w-4 h-4 mr-2" />
                        New Analysis
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Bottom CTA */}
          <div className="mt-12 text-center animate-fade-in">
            <Link href="/phenotypes">
              <Card className="p-8 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 cursor-pointer border-2 hover:border-primary/50 inline-block">
                <div className="flex items-center space-x-4">
                  <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Sparkles className="h-6 w-6 text-primary" />
                  </div>
                  <div className="text-left">
                    <h3 className="text-xl font-bold mb-1">Explore All Phenotypes</h3>
                    <p className="text-muted-foreground text-sm">
                      Browse the complete gallery of reference phenotypes
                    </p>
                  </div>
                </div>
              </Card>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
