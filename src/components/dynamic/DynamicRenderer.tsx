import React from 'react';
import { ComparisonTable } from './ComparisonTable';
import { KanbanBoard } from './KanbanBoard';
import { ScheduleCalendar } from './ScheduleCalendar';

interface DynamicRendererProps {
  type: string;
  data: any;
}

export const DynamicRenderer: React.FC<DynamicRendererProps> = ({ type, data }) => {
  switch (type) {
    case 'table':
    case 'comparison':
      return <ComparisonTable data={data} />;
    case 'kanban':
    case 'board':
      return <KanbanBoard data={data} />;
    case 'calendar':
    case 'schedule':
      return <ScheduleCalendar data={data} />;
    default:
      return (
        <div className="bg-red-50 text-red-600 p-4 rounded-lg border border-red-200">
          Unknown component type: {type}
        </div>
      );
  }
};
