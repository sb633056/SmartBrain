// src/components/DataTable.tsx
import React from 'react'

interface DataTableProps {
  data: any[];
  columns: string[];
  title: string;
}

export default function DataTable({ data, columns, title }: DataTableProps) {
  if (!data || data.length === 0) {
    return <p className="text-gray-500">No data available for {title}.</p>;
  }

  // Helper function to format column names for display (e.g., 'days_of_cover' -> 'Days of Cover')
  const formatHeader = (s: string) => {
    return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  };

  return (
    <div className="overflow-x-auto rounded-lg shadow-md border">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((colKey) => (
              <th
                key={colKey}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {formatHeader(colKey)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((row, rowIndex) => (
            <tr key={rowIndex} className="hover:bg-gray-50">
              {columns.map((colKey) => (
                <td key={colKey} className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">
                  {/* Simple formatting for numbers/strings */}
                  {typeof row[colKey] === 'number' ? row[colKey].toLocaleString('en-IN', { maximumFractionDigits: 2 }) : String(row[colKey])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
