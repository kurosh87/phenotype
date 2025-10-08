# Phenotype Matcher - Full Documentation

## 🎯 Overview

Phenotype Matcher is a Next.js application that uses AI-powered image analysis to match user-uploaded facial photos against a reference database of anthropological phenotypes. The app generates detailed reports using vector similarity search and OpenAI's GPT-4.

## 🚀 Live Application

**URL**: http://localhost:3002

## ✨ Features

### Core Functionality
- ✅ **Photo Upload** - Drag-and-drop or click to upload facial photos
- ✅ **AI Analysis** - Generate image embeddings using Replicate's CLIP model
- ✅ **Similarity Matching** - PostgreSQL pgvector for fast similarity search
- ✅ **AI Reports** - OpenAI GPT-4 generates educational analysis reports
- ✅ **User Dashboard** - View past analyses and results
- ✅ **Admin Interface** - Add and manage reference phenotypes
- ✅ **Phenotype Gallery** - Browse all reference phenotypes

### Technical Stack
- **Framework**: Next.js 15.5.4 (App Router)
- **Database**: Neon Serverless PostgreSQL with pgvector extension
- **Authentication**: Stack Auth (Neon Auth)
- **Storage**: Vercel Blob for images
- **AI Models**:
  - Replicate CLIP for image embeddings
  - OpenAI GPT-4 for report generation
- **UI**: Tailwind CSS + shadcn/ui components
- **ORM**: Drizzle ORM

## 📁 Project Structure

```
app/
├── page.tsx                      # Landing page with photo uploader
├── dashboard/page.tsx            # User dashboard
├── analysis/[id]/page.tsx        # Analysis results page
├── phenotypes/page.tsx           # Phenotype gallery
├── admin/phenotypes/page.tsx     # Admin interface
├── api/
│   ├── upload/route.ts          # Upload images to Vercel Blob
│   ├── analyze/route.ts         # Analyze photos & generate reports
│   ├── history/route.ts         # Get user's analysis history
│   └── admin/phenotypes/route.ts # Manage phenotypes (CRUD)
├── schema/schema.ts             # Drizzle database schema
└── handler/[...stack]/page.tsx  # Auth routes

components/
├── photo-uploader.tsx           # Photo upload component
├── admin-phenotype-form.tsx     # Admin form for adding phenotypes
└── ui/                          # shadcn/ui components

lib/
├── embeddings.ts                # CLIP embedding generation
├── database.ts                  # Database queries & vector search
└── utils.ts                     # Utility functions
```

## 🗄️ Database Schema

### Tables

#### `phenotypes` - Reference phenotype database
```sql
- id (uuid, primary key)
- name (text) - Name of the phenotype
- description (text) - Detailed description
- geographic_origin (text) - Geographic region
- key_traits (jsonb) - Array of characteristic traits
- image_url (text) - Vercel Blob URL
- embedding (vector[512]) - CLIP image embedding
- metadata (jsonb) - Additional metadata
- created_at (timestamp)
```

#### `user_uploads` - User uploaded photos
```sql
- id (uuid, primary key)
- user_id (text) - Stack Auth user ID
- image_url (text) - Vercel Blob URL
- embedding (vector[512]) - CLIP image embedding
- analysis_results (jsonb) - Cached results
- created_at (timestamp)
```

#### `analysis_history` - Analysis results
```sql
- id (uuid, primary key)
- user_id (text) - Stack Auth user ID
- upload_id (uuid) - Foreign key to user_uploads
- top_matches (jsonb) - Top phenotype matches
- ai_report (text) - Generated analysis report
- created_at (timestamp)
```

## 🔧 Environment Variables

```env
# Neon Auth (Stack Auth)
NEXT_PUBLIC_STACK_PROJECT_ID='your-project-id'
NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY='pck_...'
STACK_SECRET_SERVER_KEY='ssk_...'

# Database
DATABASE_URL='postgresql://...'

# Vercel Blob Storage
BLOB_READ_WRITE_TOKEN='vercel_blob_rw_...'

# AI APIs
REPLICATE_API_TOKEN='r8_...'
OPENAI_API_KEY='sk-proj-...'
AI_GATEWAY_API_KEY='vck_...'
```

## 🚦 Getting Started

### Prerequisites
- Node.js 18+
- Neon PostgreSQL account
- Vercel account (for Blob storage)
- Replicate API account
- OpenAI API account

### Installation & Setup

1. **Clone and Install**
```bash
cd neon-auth-demo-app
npm install
```

2. **Database Setup**
The database is already configured with:
- pgvector extension enabled
- All tables created via Drizzle migrations
- Vector indexes for fast similarity search

3. **Environment Configuration**
All environment variables are already set in `.env.local`

4. **Start Development Server**
```bash
npm run dev
```

Access the app at: http://localhost:3002

## 📖 User Guide

### For End Users

#### 1. Sign Up / Sign In
- Visit http://localhost:3002
- Click "Sign Up" or "Sign In"
- Create an account or log in

#### 2. Upload & Analyze a Photo
- On the homepage, drag-and-drop or click to upload a facial photo
- Supported formats: JPG, PNG (max 5MB)
- Click "Analyze Photo"
- Wait 10-30 seconds for analysis (embedding generation + AI report)

#### 3. View Results
- Automatically redirected to results page
- See top phenotype matches with similarity percentages
- Read AI-generated analysis report
- View detailed match information

#### 4. Dashboard
- Click "View History" from homepage
- See all past analyses
- Click any analysis to view details
- Track your analysis count and monthly stats

#### 5. Browse Phenotypes
- Click "Browse Phenotypes" from homepage
- View all reference phenotypes in the database
- See geographic origins and key traits

### For Administrators

#### Adding Reference Phenotypes

1. **Access Admin Panel**
   - Visit: http://localhost:3002/admin/phenotypes
   - (Note: Currently no admin role check - all authenticated users can access)

2. **Add a Phenotype**
   - Upload a clear reference image
   - Enter phenotype name (required)
   - Add description (recommended)
   - Specify geographic origin (optional)
   - List key traits, comma-separated (optional)
   - Click "Add Phenotype"

3. **Wait for Processing**
   - Image uploads to Vercel Blob
   - CLIP model generates 512-dimensional embedding
   - Phenotype saved to database with embedding
   - Process takes 5-15 seconds

4. **Verify Addition**
   - New phenotype appears in right column
   - Check phenotypes gallery to confirm

## 🔍 How It Works

### Image Analysis Pipeline

1. **Photo Upload**
   - User uploads photo → Vercel Blob storage
   - Returns public URL

2. **Embedding Generation**
   - Photo URL sent to Replicate CLIP model
   - Generates 512-dimensional vector embedding
   - Embedding represents visual features

3. **Similarity Search**
   - Query database using pgvector's cosine distance
   - Finds top 10 most similar phenotypes
   - Sorted by similarity score (0-1)

4. **AI Report Generation**
   - Top 5 matches sent to OpenAI GPT-4
   - Prompt includes phenotype names, descriptions, origins
   - GPT-4 generates educational analysis (200-300 words)

5. **Save & Display**
   - Results saved to `analysis_history` table
   - User redirected to results page
   - Can view anytime from dashboard

### Vector Similarity Search

The app uses PostgreSQL's pgvector extension for efficient similarity search:

```sql
SELECT
  id, name, image_url,
  1 - (embedding <=> query_embedding) as similarity
FROM phenotypes
WHERE embedding IS NOT NULL
ORDER BY embedding <=> query_embedding
LIMIT 10
```

- `<=>` is cosine distance operator
- `1 - distance` converts to similarity (0-1 range)
- Results sorted by closest match first

## 🎨 UI Components

### Key Components Built

1. **PhotoUploader** ([components/photo-uploader.tsx](components/photo-uploader.tsx:1))
   - Drag-and-drop interface
   - Image preview
   - Upload progress bar
   - Error handling

2. **AdminPhenotypeForm** ([components/admin-phenotype-form.tsx](components/admin-phenotype-form.tsx:1))
   - Form for adding phenotypes
   - File upload with preview
   - Progress indicators
   - Success/error messages

3. **shadcn/ui Components Used**
   - Button, Card, Input, Badge, Progress
   - Dialog, Form, Table, Tabs
   - Dropdown Menu, Avatar, Label, Separator

## 🔐 Security Considerations

### Current Implementation
- ✅ User authentication via Stack Auth
- ✅ Auth-protected routes (redirect to sign-in)
- ✅ User-specific data isolation (user_id checks)
- ✅ File type validation (images only)
- ✅ File size limits (5MB max)

### Production Recommendations
- ⚠️ **Admin Role Check**: Currently no admin role verification
  - Add role-based access control
  - Restrict `/admin/*` routes to admin users only

- ⚠️ **Rate Limiting**: No rate limits on analysis endpoint
  - Implement per-user daily analysis limits
  - Add API rate limiting middleware

- ⚠️ **API Key Rotation**: API keys in `.env.local`
  - Rotate keys regularly
  - Use environment variables in production

- ⚠️ **Content Moderation**: No image content filtering
  - Add content moderation before storage
  - Implement NSFW detection

## 🚀 Deployment

### Vercel Deployment (Recommended)

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Connect to Vercel**
- Import project on Vercel dashboard
- Connect your GitHub repository

3. **Configure Environment Variables**
- Add all variables from `.env.local` to Vercel
- Set `NEXT_PUBLIC_SITE_URL` to your Vercel domain

4. **Deploy**
- Vercel auto-deploys on git push
- Database already configured (Neon)
- Blob storage auto-configured (Vercel)

## 📊 API Endpoints

### Public Endpoints

#### `POST /api/upload`
Upload image to Vercel Blob
- **Auth**: Required
- **Body**: `FormData` with `file`
- **Response**: `{ url: string }`

#### `POST /api/analyze`
Analyze uploaded photo
- **Auth**: Required
- **Body**: `{ imageUrl: string }`
- **Response**: `{ analysisId, matches, aiReport, uploadedImageUrl }`

#### `GET /api/history`
Get user's analysis history
- **Auth**: Required
- **Query**: `?limit=20`
- **Response**: `{ history: Array }`

### Admin Endpoints

#### `POST /api/admin/phenotypes`
Add new phenotype
- **Auth**: Required (admin check TODO)
- **Body**: `FormData` with file, name, description, etc.
- **Response**: `{ success: true, phenotype }`

#### `GET /api/admin/phenotypes`
List all phenotypes
- **Auth**: Required
- **Response**: `{ phenotypes: Array }`

## 🐛 Troubleshooting

### Common Issues

**Issue**: "No phenotypes found in database"
- **Solution**: Go to `/admin/phenotypes` and add reference phenotypes first

**Issue**: "Failed to generate embedding"
- **Solution**: Check `REPLICATE_API_TOKEN` is set correctly

**Issue**: "Analysis timeout"
- **Solution**: CLIP embedding generation can take 10-30 seconds. Wait patiently.

**Issue**: "No matches found"
- **Solution**: Ensure database has phenotypes with embeddings

## 📈 Performance Optimization

### Current Performance
- Image upload: ~2-5 seconds
- Embedding generation: ~10-30 seconds (Replicate API)
- Similarity search: ~50-200ms (pgvector)
- AI report generation: ~3-10 seconds (OpenAI GPT-4)
- **Total analysis time**: ~15-45 seconds

### Optimization Tips
1. **Use pgvector IVFFlat index** for faster searches (100+ phenotypes)
2. **Cache embeddings** to avoid regeneration
3. **Batch process** multiple analyses
4. **Use GPT-4o-mini** for faster report generation
5. **CDN for images** via Vercel Blob automatic CDN

## 🎯 Future Enhancements

### V2 Features (Not Implemented)
- [ ] Bulk photo upload
- [ ] Photo comparison tool (compare two people)
- [ ] PDF report export
- [ ] Social sharing of results
- [ ] Public/private analysis toggle
- [ ] Favorite/bookmark results
- [ ] Admin dashboard with analytics
- [ ] Proper admin role management
- [ ] Rate limiting and usage quotas
- [ ] Email notifications
- [ ] API access for developers
- [ ] Premium features with Stripe integration

## 📝 Development Notes

### Database Migrations
```bash
# Generate new migration
npm run drizzle:generate

# Apply migrations
npm run drizzle:migrate
```

### Adding New shadcn Components
```bash
npx shadcn@latest add [component-name]
```

### Code Formatting
```bash
npm run format
```

## 👥 Credits

Built using:
- Next.js 15
- Neon PostgreSQL
- Stack Auth (Neon Auth)
- Vercel Blob
- Replicate AI
- OpenAI
- Drizzle ORM
- shadcn/ui
- Tailwind CSS

## 📄 License

This is a demo application. Modify and use as needed.

---

**Need Help?** Check the code comments or contact the development team.
