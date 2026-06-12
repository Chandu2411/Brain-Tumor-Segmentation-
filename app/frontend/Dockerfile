# Use official Node image
FROM node:22-alpine

# Install pnpm
RUN npm install -g pnpm@11.6.0

# Set working directory
WORKDIR /app

# Copy dependency configs
COPY package.json pnpm-lock.yaml* ./

# Install dependencies
RUN pnpm install

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Run dev server
CMD ["pnpm", "dev"]
