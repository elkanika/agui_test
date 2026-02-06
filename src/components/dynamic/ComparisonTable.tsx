import React from 'react';

interface ComparisonTableProps {
  data: {
    headers: string[];
    rows: string[][];
  };
}

export const ComparisonTable: React.FC<ComparisonTableProps> = ({ data }) => {
  if (!data || !data.headers || !data.rows) return <div>Invalid Table Data</div>;

  return (
    <div className="overflow-x-auto my-4 rounded-lg border border-gray-200 shadow-sm">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {data.headers.map((header, idx) => (
              <th
                key={idx}
                scope="col"
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.rows.map((row, rowIdx) => (
            <tr key={rowIdx}>
              {row.map((cell, cellIdx) => (
                <td
                  key={cellIdx}
                  className="px-6 py-4 whitespace-pre-wrap text-sm text-gray-900"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
