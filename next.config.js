/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "img.clerk.com",
        port: "",
      },
      {
        protocol: "https",
        hostname: "7ku24g8oti06nmwx.public.blob.vercel-storage.com",
        port: "",
      },
    ],
  },
  webpack: (config) => {
    config.externals.push("bun:sqlite");
    return config;
  },
};

module.exports = nextConfig;
