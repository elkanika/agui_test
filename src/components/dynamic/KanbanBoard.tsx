import React from 'react';

interface KanbanBoardProps {
  data: {
    columns: {
      title: string;
      items: string[];
    }[];
  };
}

export const KanbanBoard: React.FC<KanbanBoardProps> = ({ data }) => {
  if (!data || !data.columns) return <div>Invalid Kanban Data</div>;

  return (
    <div className="flex overflow-x-auto gap-4 my-4 pb-4">
      {data.columns.map((col, colIdx) => (
        <div
          key={colIdx}
          className="min-w-[250px] bg-gray-100 rounded-lg p-3 flex flex-col max-h-[500px]"
        >
          <h3 className="font-semibold text-gray-700 mb-3 px-1">{col.title}</h3>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {col.items.map((item, itemIdx) => (
              <div
                key={itemIdx}
                className="bg-white p-3 rounded shadow-sm border border-gray-200 text-sm text-gray-800"
              >
                {item}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};
