# Database Configuration

## Using Existing Table

The project works with your existing `insurance_data` table. Just ensure:

1. **Table exists** in your PostgreSQL database
2. **Columns match** the schema in `config/schema.yaml`
3. **Environment variables** point to your database:

```env
POSTGRES_URL=postgresql://username:password@localhost:5432/your_database