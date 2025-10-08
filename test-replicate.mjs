import { config } from "dotenv";
config({ path: ".env.local" });

import Replicate from "replicate";

console.log("Token from env:", process.env.REPLICATE_API_TOKEN);

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

console.log("Replicate instance created");

try {
  const output = await replicate.run(
    "andreasjansson/clip-features:75b33f253f7714a281ad3e9b28f63e3232d583716ef6718f2e46641077ea040a",
    {
      input: {
        inputs: "https://7ku24g8oti06nmwx.public.blob.vercel-storage.com/phenotypes/Ainuid/male_1759937963381.jpg",
      },
    }
  );
  console.log("Success! Got embedding:", output);
} catch (error) {
  console.error("Error:", error.message);
}
