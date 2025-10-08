import { stackServerApp } from "app/stack";
import { ModernHeader } from "@/components/modern-header";
import { PhotoUploader } from "@/components/photo-uploader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import Link from "next/link";
import { Sparkles, Zap, Shield, TrendingUp } from "lucide-react";

export default async function Home() {
  const user = await stackServerApp.getUser();
  const userData = user ? {
    displayName: user.displayName,
    primaryEmail: user.primaryEmail,
  } : null;

  return (
    <div className="min-h-screen flex flex-col">
      <ModernHeader user={userData} />

      <main className="flex-1">
        {user ? (
          <>
            {/* Hero Section for Logged In Users */}
            <section className="gradient-mesh py-16 md:py-24">
              <div className="container px-4 md:px-6">
                <div className="flex flex-col items-center space-y-4 text-center animate-fade-in">
                  <div className="inline-flex items-center rounded-full border px-4 py-1.5 text-sm font-medium bg-primary/10 text-primary">
                    <Sparkles className="mr-2 h-4 w-4" />
                    AI-Powered Phenotype Analysis
                  </div>
                  <div className="space-y-4 max-w-3xl">
                    <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl text-balance">
                      Discover Your
                      <span className="text-primary"> Phenotype Match</span>
                    </h1>
                    <p className="mx-auto max-w-[700px] text-muted-foreground text-lg md:text-xl">
                      Upload your photo and let our advanced AI analyze and match you
                      with anthropological phenotypes from around the world.
                    </p>
                  </div>
                </div>

                {/* Upload Section */}
                <div className="mt-16 animate-slide-up">
                  <PhotoUploader />
                </div>

                {/* Quick Stats */}
                <div className="mt-16 grid gap-8 md:grid-cols-3">
                  <Card className="p-6 text-center hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
                    <div className="flex justify-center mb-4">
                      <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                        <Zap className="h-6 w-6 text-primary" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold mb-2">Instant Analysis</h3>
                    <p className="text-muted-foreground">
                      Get detailed results in under 30 seconds
                    </p>
                  </Card>

                  <Card className="p-6 text-center hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
                    <div className="flex justify-center mb-4">
                      <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                        <TrendingUp className="h-6 w-6 text-primary" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold mb-2">AI-Powered</h3>
                    <p className="text-muted-foreground">
                      Advanced embeddings for accurate matching
                    </p>
                  </Card>

                  <Card className="p-6 text-center hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
                    <div className="flex justify-center mb-4">
                      <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                        <Shield className="h-6 w-6 text-primary" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold mb-2">Private & Secure</h3>
                    <p className="text-muted-foreground">
                      Your data is encrypted and protected
                    </p>
                  </Card>
                </div>

                {/* CTA Cards */}
                <div className="mt-16 grid gap-6 md:grid-cols-2 max-w-4xl mx-auto">
                  <Link href="/dashboard">
                    <Card className="p-8 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 cursor-pointer border-2 hover:border-primary/50">
                      <div className="flex items-start space-x-4">
                        <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                          <TrendingUp className="h-6 w-6 text-primary" />
                        </div>
                        <div className="space-y-2">
                          <h3 className="text-xl font-bold">View History</h3>
                          <p className="text-muted-foreground text-sm">
                            See all your past analyses and detailed reports
                          </p>
                        </div>
                      </div>
                    </Card>
                  </Link>

                  <Link href="/phenotypes">
                    <Card className="p-8 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 cursor-pointer border-2 hover:border-primary/50">
                      <div className="flex items-start space-x-4">
                        <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                          <Sparkles className="h-6 w-6 text-primary" />
                        </div>
                        <div className="space-y-2">
                          <h3 className="text-xl font-bold">Browse Gallery</h3>
                          <p className="text-muted-foreground text-sm">
                            Explore the complete database of phenotypes
                          </p>
                        </div>
                      </div>
                    </Card>
                  </Link>
                </div>
              </div>
            </section>
          </>
        ) : (
          <>
            {/* Hero Section for Non-Logged In Users */}
            <section className="gradient-mesh py-20 md:py-32">
              <div className="container px-4 md:px-6">
                <div className="flex flex-col items-center space-y-8 text-center">
                  <div className="space-y-6 max-w-4xl animate-fade-in">
                    <div className="inline-flex items-center rounded-full border px-4 py-1.5 text-sm font-medium bg-primary/10 text-primary">
                      <Sparkles className="mr-2 h-4 w-4" />
                      Powered by Advanced AI
                    </div>
                    <h1 className="text-5xl font-bold tracking-tighter sm:text-6xl md:text-7xl lg:text-8xl text-balance">
                      Discover Your
                      <br />
                      <span className="text-primary">Phenotype Identity</span>
                    </h1>
                    <p className="mx-auto max-w-[700px] text-muted-foreground text-xl md:text-2xl">
                      Upload your photo and match with anthropological phenotypes
                      using state-of-the-art AI technology.
                    </p>
                  </div>

                  <div className="flex flex-col sm:flex-row gap-4 animate-slide-up">
                    <Link href="/handler/sign-up">
                      <Button size="lg" className="h-12 px-8 text-base">
                        Get Started Free
                      </Button>
                    </Link>
                    <Link href="/handler/sign-in">
                      <Button size="lg" variant="outline" className="h-12 px-8 text-base">
                        Sign In
                      </Button>
                    </Link>
                  </div>

                  {/* Demo Image */}
                  <div className="mt-12 w-full max-w-5xl mx-auto animate-slide-up">
                    <Card className="p-4 shadow-2xl">
                      <div className="aspect-video rounded-lg bg-gradient-to-br from-primary/20 via-primary/5 to-background flex items-center justify-center">
                        <div className="text-center space-y-4">
                          <Sparkles className="h-16 w-16 mx-auto text-primary/40" />
                          <p className="text-muted-foreground">
                            AI-Powered Analysis Preview
                          </p>
                        </div>
                      </div>
                    </Card>
                  </div>
                </div>
              </div>
            </section>

            {/* Features Section */}
            <section className="py-20 bg-muted/30">
              <div className="container px-4 md:px-6">
                <div className="text-center mb-16">
                  <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl mb-4">
                    Why Choose Phenotype Matcher?
                  </h2>
                  <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
                    Experience the most advanced phenotype matching technology
                    powered by AI
                  </p>
                </div>

                <div className="grid gap-8 md:grid-cols-3">
                  <Card className="p-8 text-center hover:shadow-xl transition-all duration-300 hover:-translate-y-2">
                    <div className="flex justify-center mb-6">
                      <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center">
                        <Zap className="h-8 w-8 text-primary" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold mb-3">Lightning Fast</h3>
                    <p className="text-muted-foreground">
                      Advanced image embeddings deliver results in seconds, not
                      minutes. Get instant matches with our optimized AI pipeline.
                    </p>
                  </Card>

                  <Card className="p-8 text-center hover:shadow-xl transition-all duration-300 hover:-translate-y-2">
                    <div className="flex justify-center mb-6">
                      <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center">
                        <TrendingUp className="h-8 w-8 text-primary" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold mb-3">Highly Accurate</h3>
                    <p className="text-muted-foreground">
                      Vector similarity search with 512-dimensional embeddings
                      ensures the most accurate phenotype matching available.
                    </p>
                  </Card>

                  <Card className="p-8 text-center hover:shadow-xl transition-all duration-300 hover:-translate-y-2">
                    <div className="flex justify-center mb-6">
                      <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center">
                        <Shield className="h-8 w-8 text-primary" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold mb-3">Privacy First</h3>
                    <p className="text-muted-foreground">
                      Your photos and data are encrypted and stored securely. We
                      never share your information with third parties.
                    </p>
                  </Card>
                </div>
              </div>
            </section>

            {/* CTA Section */}
            <section className="py-20 gradient-mesh">
              <div className="container px-4 md:px-6">
                <Card className="max-w-3xl mx-auto p-12 text-center shadow-2xl">
                  <div className="space-y-6">
                    <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                      Ready to Discover Your Match?
                    </h2>
                    <p className="text-muted-foreground text-lg">
                      Join thousands of users exploring their phenotype identity
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                      <Link href="/handler/sign-up">
                        <Button size="lg" className="h-12 px-8 text-base">
                          Start Free Analysis
                        </Button>
                      </Link>
                    </div>
                  </div>
                </Card>
              </div>
            </section>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t py-8">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-sm text-muted-foreground">
              © 2025 Phenotype Matcher. Powered by AI.
            </p>
            <div className="flex gap-6">
              <Link
                href="/phenotypes"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Gallery
              </Link>
              <Link
                href="/dashboard"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Dashboard
              </Link>
              <Link
                href="/admin/phenotypes"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Admin
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
